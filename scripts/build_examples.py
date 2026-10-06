"""Build the example gallery and the Colab notebooks from the example scripts.

Each `examples/NN_name.py` is a jupytext percent-format script, with cells that
start with `# %%`, and it's the only source of its gallery page and notebook.
For each script, this tool:

1. Converts the script to a notebook and executes it in a scratch directory,
   unless the cache already holds an executed copy with the same key. The key
   covers the script, the jbubble version, the kernelspec, the files in the
   jbubble package, and `uv.lock`.
2. Writes a page with the code, the printed output, and the figures to
   `docs/examples/NN_name.md`, plus a gallery of thumbnails to
   `docs/examples/index.md`. Each page ends with links to the previous and
   the next example and to the gallery.
3. Writes the executed notebook, with an install cell before the first code
   cell, to `docs/examples/notebooks/NN_name.ipynb`. The built site serves it
   at `examples/notebooks/NN_name.ipynb`. CI publishes the site to the
   `gh-pages` branch, and each page's Colab badge opens the notebook there.

The build fails when a script has no `# %%` cells, when a cell raises an
exception, or when an executed example shows no PNG figure.

To see the options, run `uv run python scripts/build_examples.py --help`.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass

import jupytext
import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = "jbubble-examples"
# Marks an output directory as generated, so that the tool only ever deletes
# a directory that it wrote.
MARKER = ".generated-by-build-examples"
IMAGE_TYPES = {"image/png": "png", "image/gif": "gif", "image/svg+xml": "svg"}
# Versions that PyPI can serve, such as 0.2.0 or 0.2.0rc1, but not the
# development versions between releases, such as 0.2.0.post1.dev3.
RELEASE = re.compile(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
INTRO = """\
Each example is a Python script in the
[`examples/`](https://github.com/{repo}/tree/{ref}/examples) directory of the
repository. Its page shows the code with the output that it produced when the
site was built. To run an example yourself, open it in Google Colab, download
the notebook, or run the script with `python`.
"""


class ExampleError(Exception):
    """An example that can't become a gallery page."""


@dataclass(frozen=True)
class Settings:
    out: pathlib.Path
    cache_dir: pathlib.Path
    package_dir: pathlib.Path
    execute: bool
    timeout: int
    repo: str
    source_ref: str
    notebooks_ref: str
    notebooks_path: str
    pip_spec: str
    version: str


@dataclass(frozen=True)
class Entry:
    name: str
    title: str
    summary: str
    thumb: str | None
    status: str
    seconds: float
    page: str


def default_pip_spec(version: str, repo: str, ref: str) -> str:
    """Pin a released version; install anything else from git at `ref`.

    The `io` extra installs h5py, which the parameter sweep example needs
    outside Colab.
    """
    if RELEASE.fullmatch(version):
        return f"jbubble[examples,io]=={version}"
    return f"jbubble[examples,io] @ git+https://github.com/{repo}@{ref}"


def cache_key(script: pathlib.Path, settings: Settings) -> str:
    """Hash everything that can change an example's output."""
    h = hashlib.sha256()
    h.update(script.read_bytes())
    h.update(settings.version.encode())
    h.update(json.dumps(kernel_spec(), sort_keys=True).encode())
    lock = ROOT / "uv.lock"
    if lock.exists():
        h.update(lock.read_bytes())
    for path in sorted(settings.package_dir.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            h.update(path.relative_to(settings.package_dir).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def kernel_spec() -> dict:
    """Return a kernelspec that runs this interpreter.

    Without it, a user-level `python3` kernelspec could run the examples in
    another environment. The spec also forces Matplotlib's inline backend: an
    inherited `MPLBACKEND=Agg` would otherwise drop every figure silently.
    The backend draws each figure at twice its display size (the "retina"
    format), so figures stay sharp on high-density screens.
    """
    return {
        "argv": [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
            "--InlineBackend.figure_formats=retina",
        ],
        "display_name": KERNEL,
        "language": "python",
        "env": {"MPLBACKEND": "module://matplotlib_inline.backend_inline"},
    }


@contextlib.contextmanager
def private_kernel() -> Iterator[None]:
    """Register the kernelspec from `kernel_spec` while the block runs.

    Jupyter finds the spec through `JUPYTER_PATH`, which this function
    restores afterwards.
    """
    previous = os.environ.get("JUPYTER_PATH")
    with tempfile.TemporaryDirectory() as directory:
        spec = pathlib.Path(directory) / "kernels" / KERNEL
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps(kernel_spec()))
        os.environ["JUPYTER_PATH"] = os.pathsep.join(
            [directory, *filter(None, [previous])]
        )
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous


def execute(nb: nbformat.NotebookNode, timeout: int) -> nbformat.NotebookNode:
    from nbclient import NotebookClient

    with tempfile.TemporaryDirectory() as cwd:
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name=KERNEL,
            record_timing=False,
            resources={"metadata": {"path": cwd}},
        )
        client.execute()
    return nb


def install_cell(pip_spec: str) -> nbformat.NotebookNode:
    # The guard keeps "Run all" in a local environment from replacing an
    # existing jbubble install, such as an editable one.
    cell = nbformat.v4.new_code_cell(
        "# Install jbubble on Google Colab, or anywhere it's missing.\n"
        "import importlib.util\n\n"
        'if importlib.util.find_spec("jbubble") is None:\n'
        f'    %pip install --quiet "{pip_spec}"'
    )
    cell.metadata["tags"] = ["install"]
    return cell


def published_notebook(
    nb: nbformat.NotebookNode, pip_spec: str
) -> nbformat.NotebookNode:
    """Copy an executed notebook and add the install cell for Colab."""
    out = nbformat.from_dict(json.loads(json.dumps(nb)))
    out.metadata.pop("jupytext", None)
    out.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    first_code = next(
        (i for i, cell in enumerate(out.cells) if cell.cell_type == "code"),
        len(out.cells),
    )
    out.cells.insert(first_code, install_cell(pip_spec))
    return out


def fenced(text: str, info: str) -> str:
    """Fence a block with more backticks than any run inside it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{info}\n{text}\n{fence}"


def output_block(text: str) -> str:
    # docs/stylesheets/extra.css styles .jb-output apart from the code.
    return fenced(text, "{ .text .jb-output }")


def title_and_summary(nb: nbformat.NotebookNode, fallback: str) -> tuple[str, str]:
    first = next((c for c in nb.cells if c.cell_type == "markdown"), None)
    if first is None:
        return fallback, ""
    lines = first.source.strip().splitlines()
    if not lines or not lines[0].startswith("# "):
        return fallback, ""
    body = "\n".join(lines[1:]).strip().split("\n\n")[0]
    return lines[0][2:].strip(), " ".join(body.split())


def to_markdown(
    nb: nbformat.NotebookNode, name: str, title: str, links: str, media: pathlib.Path
) -> tuple[str, str | None]:
    """Render a notebook as a gallery page; return the page and its thumbnail."""
    parts: list[str] = []
    thumb = None
    n_img = 0
    heading = title
    cells = list(nb.cells)
    first = cells[0] if cells else None
    if first is not None and first.cell_type == "markdown":
        head, _, rest = first.source.strip().partition("\n")
        if head.startswith("# "):
            parts.extend([head, links, rest.strip()])
            cells = cells[1:]
    if not parts:
        parts.extend([f"# {title}", links])

    for cell in cells:
        if cell.cell_type == "markdown":
            # The nearest heading is the alt text for the figures that follow.
            for line in cell.source.splitlines():
                if line.startswith("#"):
                    heading = line.lstrip("#").strip()
            parts.append(cell.source)
            continue
        if cell.cell_type != "code":
            continue
        # Cells marked active="ipynb" are commented out in the script, so the
        # page shows their output but not their code.
        if cell.metadata.get("active") != "ipynb":
            parts.append(fenced(cell.source, "python"))
        # The kernel sends printed text in chunks as it arrives, so join
        # consecutive chunks into one block, as Jupyter shows them.
        printed: list[str] = []
        for out in [*cell.get("outputs", []), None]:
            if out is not None and out.output_type == "stream":
                if out.name == "stdout":
                    printed.append(out.text)
                continue
            if printed:
                parts.append(output_block(ANSI.sub("", "".join(printed)).rstrip()))
                printed = []
            if out is None:
                break
            if out.output_type in ("display_data", "execute_result"):
                mime = next((m for m in IMAGE_TYPES if m in out.data), None)
                if mime is None:
                    if out.output_type == "execute_result" and "text/plain" in out.data:
                        parts.append(output_block(out.data["text/plain"].rstrip()))
                    continue
                n_img += 1
                fname = f"{name}_{n_img}.{IMAGE_TYPES[mime]}"
                data = out.data[mime]
                payload = (
                    data.encode() if mime == "image/svg+xml" else base64.b64decode(data)
                )
                (media / fname).write_bytes(payload)
                parts.append(f"![{heading}](media/{fname}){image_size(out, mime)}")
                # The first PNG is the thumbnail, unless a later cell is tagged
                # "thumbnail" (in the script: `# %% tags=["thumbnail"]`).
                tagged = "thumbnail" in cell.metadata.get("tags", [])
                if mime == "image/png" and (thumb is None or tagged):
                    thumb = fname
            elif out.output_type == "error":
                raise ExampleError(f"{name}: {out.ename}: {out.evalue}")
    return "\n\n".join(p for p in parts if p) + "\n", thumb


def image_size(out: nbformat.NotebookNode, mime: str) -> str:
    """Return the display size of an image output as Markdown attributes.

    A retina figure's PNG has twice the pixels of its display size, which
    the output's metadata records. Without the attributes, the page would
    show the figure at twice its size, up to the width of the page.
    """
    size = out.get("metadata", {}).get(mime, {})
    attrs = [f'{key}="{size[key]}"' for key in ("width", "height") if key in size]
    return "{ " + " ".join(attrs) + " }" if attrs else ""


def source_path(script: pathlib.Path) -> str:
    """Return the script's path in the repository."""
    try:
        return script.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"examples/{script.name}"


def link_bar(name: str, script: pathlib.Path, settings: Settings) -> str:
    colab = (
        f"https://colab.research.google.com/github/{settings.repo}/blob/"
        f"{settings.notebooks_ref}/{settings.notebooks_path}/{name}.ipynb"
    )
    source = (
        f"https://github.com/{settings.repo}/blob/{settings.source_ref}/"
        f"{source_path(script)}"
    )
    return (
        '<div class="jb-example-links" markdown="span">\n'
        f"[![Open in Colab]({COLAB_BADGE})]({colab})\n"
        f'[Download notebook](notebooks/{name}.ipynb){{ .md-button download="{name}.ipynb" }}\n'
        f"[View source]({source}){{ .md-button }}\n"
        "</div>"
    )


def build_one(script: pathlib.Path, settings: Settings) -> Entry:
    name = script.stem
    if "\n# %%" not in "\n" + script.read_text():
        # Without explicit cells, jupytext splits the script at blank lines,
        # and a figure created in one cell and drawn in the next renders empty.
        raise ExampleError(f"{script}: split the script into '# %%' cells.")
    key = cache_key(script, settings)
    cache = settings.cache_dir / f"{name}-{key}.ipynb"
    start = time.perf_counter()
    if cache.exists():
        nb = nbformat.read(cache, as_version=4)
        status = "cached"
    else:
        nb = jupytext.read(script)
        if settings.execute:
            nb = execute(nb, settings.timeout)
            settings.cache_dir.mkdir(parents=True, exist_ok=True)
            for stale in settings.cache_dir.glob(f"{name}-*.ipynb"):
                stale.unlink()
            nbformat.write(nb, cache)
            status = "executed"
        else:
            status = "not executed"
    seconds = time.perf_counter() - start

    title, summary = title_and_summary(nb, name)
    links = link_bar(name, script, settings)
    page, thumb = to_markdown(nb, name, title, links, settings.out / "media")
    if settings.execute and thumb is None:
        raise ExampleError(f"{script}: the example shows no PNG figure.")
    nbformat.write(
        published_notebook(nb, settings.pip_spec),
        settings.out / "notebooks" / f"{name}.ipynb",
    )
    return Entry(name, title, summary, thumb, status, seconds, page)


def link_text(text: str) -> str:
    return text.replace("[", r"\[").replace("]", r"\]")


def page_nav(previous: Entry | None, following: Entry | None) -> str:
    """Link an example page to its neighbours and to the gallery.

    The site's navigation lists only the gallery, so the theme shows no
    previous and next links on an example page.
    """
    links = []
    if previous is not None:
        title = link_text(previous.title)
        links.append(f"[Previous: {title}]({previous.name}.md){{ .jb-previous }}")
    links.append("[All examples](index.md){ .jb-gallery }")
    if following is not None:
        title = link_text(following.title)
        links.append(f"[Next: {title}]({following.name}.md){{ .jb-next }}")
    # docs/stylesheets/extra.css lays out .jb-example-nav and its links.
    return (
        '<nav class="jb-example-nav" aria-label="Examples" markdown="span">\n'
        + "\n".join(links)
        + "\n</nav>\n"
    )


def write_pages(entries: list[Entry], settings: Settings) -> None:
    """Write the page of each example, in gallery order."""
    for i, entry in enumerate(entries):
        previous = entries[i - 1] if i > 0 else None
        following = entries[i + 1] if i + 1 < len(entries) else None
        (settings.out / f"{entry.name}.md").write_text(
            entry.page + "\n" + page_nav(previous, following)
        )


def write_index(entries: list[Entry], settings: Settings) -> None:
    cards = []
    for e in entries:
        title = link_text(e.title)
        image = f"[![{title}](media/{e.thumb})]({e.name}.md)\n\n    " if e.thumb else ""
        summary = f"\n\n    {e.summary}" if e.summary else ""
        cards.append(f"-   {image}**[{title}]({e.name}.md)**{summary}\n")
    intro = INTRO.format(repo=settings.repo, ref=settings.source_ref)
    (settings.out / "index.md").write_text(
        "# Examples\n\n"
        + intro
        + '\n<div class="grid cards" markdown>\n\n'
        + "\n".join(cards)
        + "\n</div>\n"
    )


def prepare_output(out: pathlib.Path) -> None:
    if out.exists():
        if any(out.iterdir()) and not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and wasn't written by this tool.")
        shutil.rmtree(out)
    (out / "media").mkdir(parents=True)
    (out / "notebooks").mkdir()
    (out / MARKER).touch()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--examples",
        type=pathlib.Path,
        default=ROOT / "examples",
        help="directory with the example scripts",
    )
    p.add_argument("--only", default="[0-9][0-9]_*.py", help="glob for the scripts")
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "docs" / "examples",
        help="output directory, deleted and rewritten on each run",
    )
    p.add_argument(
        "--cache-dir", type=pathlib.Path, default=ROOT / "build" / "examples-cache"
    )
    p.add_argument("--package-dir", type=pathlib.Path, default=ROOT / "jbubble")
    p.add_argument(
        "--no-execute",
        dest="execute",
        action="store_false",
        help="write the pages and notebooks without running the examples",
    )
    p.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    p.add_argument("--jobs", type=int, default=2, help="examples to run at once")
    p.add_argument("--repo", default="imperial-nsb/jbubble")
    p.add_argument(
        "--source-ref",
        default="main",
        help="git ref for the source links, and for the install line of an "
        "unreleased version",
    )
    p.add_argument(
        "--notebooks-ref",
        default="gh-pages",
        help="branch that serves the built site, for the Colab badges",
    )
    p.add_argument(
        "--notebooks-path",
        default="examples/notebooks",
        help="path of the notebooks inside the built site",
    )
    p.add_argument(
        "--pip-spec",
        help="requirement for the install cell (default: jbubble[examples,io]==VERSION "
        "for a release, otherwise a git URL at --source-ref)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    version = importlib.metadata.version("jbubble")
    settings = Settings(
        out=args.out.resolve(),
        cache_dir=args.cache_dir.resolve(),
        package_dir=args.package_dir.resolve(),
        execute=args.execute,
        timeout=args.timeout,
        repo=args.repo,
        source_ref=args.source_ref,
        notebooks_ref=args.notebooks_ref,
        notebooks_path=args.notebooks_path.strip("/"),
        pip_spec=args.pip_spec or default_pip_spec(version, args.repo, args.source_ref),
        version=version,
    )
    scripts = sorted(args.examples.glob(args.only))
    if not scripts:
        raise SystemExit(f"No scripts match {args.examples / args.only}.")

    prepare_output(settings.out)
    print(f"jbubble {version}; install cell: {settings.pip_spec}", flush=True)
    entries: list[Entry] = []
    failures: list[str] = []
    with (
        private_kernel(),
        concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool,
    ):
        futures = {pool.submit(build_one, s, settings): s for s in scripts}
        for future in concurrent.futures.as_completed(futures):
            try:
                entry = future.result()
            except ExampleError as error:
                failures.append(str(error))
                continue
            except Exception as error:  # report every failing example
                failures.append(f"{futures[future]}: {error}")
                continue
            entries.append(entry)
            print(
                f"{entry.name:<36s} {entry.status:<13s} {entry.seconds:7.1f} s",
                flush=True,
            )

    if failures:
        print(f"\n{len(failures)} example(s) failed:\n", file=sys.stderr)
        print("\n\n".join(failures), file=sys.stderr)
        return 1
    entries.sort(key=lambda e: e.name)
    write_pages(entries, settings)
    write_index(entries, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
