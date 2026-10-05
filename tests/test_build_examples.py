"""Tests for scripts/build_examples.py, which builds the example gallery.

The script needs the docs dependency group (`uv sync --group docs`), so these
tests skip themselves without it.
"""

import base64
import importlib.util
import os
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_examples.py"
if not SCRIPT_PATH.exists():  # the sdist ships tests/ but not scripts/
    pytest.skip("needs scripts/ from a repository checkout", allow_module_level=True)
pytest.importorskip("jupytext")
pytest.importorskip("nbclient")

import jupytext  # noqa: E402
import nbformat  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_examples", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
build_examples = importlib.util.module_from_spec(_spec)
sys.modules["build_examples"] = build_examples
_spec.loader.exec_module(build_examples)

# A 1x1 PNG, enough for an image output.
PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode()

SCRIPT = """\
# %% [markdown]
# # A small example
#
# Plot a line.

# %%
import matplotlib.pyplot as plt

# %% tags=["thumbnail"]
plt.plot([0, 1], [0, 1])
plt.show()
"""


def settings(tmp_path, **overrides):
    values = {
        "out": tmp_path / "out",
        "cache_dir": tmp_path / "cache",
        "package_dir": ROOT / "jbubble",
        "execute": True,
        "timeout": 120,
        "repo": "imperial-nsb/jbubble",
        "source_ref": "main",
        "notebooks_ref": "gh-pages",
        "notebooks_path": "examples/notebooks",
        "pip_spec": "jbubble[examples]==0.2.0",
        "version": "0.2.0",
    }
    values.update(overrides)
    return build_examples.Settings(**values)


@pytest.mark.parametrize("version", ["0.2.0", "0.2.0rc1", "1.0.0.post1"])
def test_released_versions_are_pinned(version):
    spec = build_examples.default_pip_spec(version, "imperial-nsb/jbubble", "abc123")
    assert spec == f"jbubble[examples]=={version}"


@pytest.mark.parametrize("version", ["0.2.0.post1.dev3", "0.0.0+unknown"])
def test_unreleased_versions_install_from_git(version):
    spec = build_examples.default_pip_spec(version, "imperial-nsb/jbubble", "abc123")
    assert spec == (
        "jbubble[examples] @ git+https://github.com/imperial-nsb/jbubble@abc123"
    )


def test_cache_key_covers_the_script_package_version_and_lock(tmp_path, monkeypatch):
    # A stale key would publish notebooks executed with old dependencies.
    monkeypatch.setattr(build_examples, "ROOT", tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text("jax 0.8.1\n")
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "core.py").write_text("x = 1\n")
    script = tmp_path / "01_small.py"
    script.write_text(SCRIPT)
    base = settings(tmp_path, package_dir=package)
    key = build_examples.cache_key(script, base)
    assert build_examples.cache_key(script, base) == key

    lock.write_text("jax 0.11.1\n")
    assert build_examples.cache_key(script, base) != key
    lock.write_text("jax 0.8.1\n")
    bumped = settings(tmp_path, package_dir=package, version="0.2.1")
    assert build_examples.cache_key(script, bumped) != key
    (package / "core.py").write_text("x = 2\n")
    assert build_examples.cache_key(script, base) != key
    (package / "core.py").write_text("x = 1\n")
    script.write_text(SCRIPT + "\n# %%\nprint(1)\n")
    assert build_examples.cache_key(script, base) != key
    script.write_text(SCRIPT)
    assert build_examples.cache_key(script, base) == key
    spec = build_examples.kernel_spec()
    spec["argv"] = spec["argv"][:-1]  # without the retina figure format
    monkeypatch.setattr(build_examples, "kernel_spec", lambda: spec)
    assert build_examples.cache_key(script, base) != key


def test_scripts_without_cells_are_refused(tmp_path):
    script = tmp_path / "01_plain.py"
    script.write_text('"""A plain script."""\n\nprint("no cells")\n')
    with pytest.raises(build_examples.ExampleError, match="# %%"):
        build_examples.build_one(script, settings(tmp_path))


def test_install_cell_comes_before_the_first_code_cell():
    nb = jupytext.reads(SCRIPT, fmt="py:percent")
    out = build_examples.published_notebook(nb, "jbubble[examples]==0.2.0")
    kinds = [cell.cell_type for cell in out.cells]
    assert kinds == ["markdown", "code", "code", "code"]
    assert out.cells[1].metadata["tags"] == ["install"]
    assert '%pip install --quiet "jbubble[examples]==0.2.0"' in out.cells[1].source
    assert out.metadata["kernelspec"]["name"] == "python3"
    assert "jupytext" not in out.metadata
    # The executed copy in the cache stays unchanged.
    assert len(nb.cells) == 3


def test_page_shows_code_output_and_figures(tmp_path):
    nb = jupytext.reads(SCRIPT, fmt="py:percent")
    nb.cells[1].outputs = [nbformat.v4.new_output("stream", text="hello\n")]
    nb.cells[2].outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={"image/png": PNG},
            metadata={"image/png": {"width": 389, "height": 196}},
        ),
        nbformat.v4.new_output(
            "execute_result", data={"text/plain": "42"}, execution_count=1
        ),
    ]
    media = tmp_path / "media"
    media.mkdir()
    page, thumb = build_examples.to_markdown(
        nb, "01_small", "A small example", "LINKS", media
    )
    assert page.startswith("# A small example\n\nLINKS\n\nPlot a line.")
    assert "```python\nimport matplotlib.pyplot as plt\n```" in page
    assert "```{ .text .jb-output }\nhello\n```" in page
    assert "```{ .text .jb-output }\n42\n```" in page
    # A retina figure shows at the display size in its metadata.
    image = '![A small example](media/01_small_1.png){ width="389" height="196" }'
    assert image in page
    assert thumb == "01_small_1.png"
    assert (media / thumb).read_bytes().startswith(b"\x89PNG")


def test_printed_chunks_share_one_output_block(tmp_path):
    # The kernel sends a loop's prints as separate chunks; stderr between
    # them doesn't split the block, but a figure does.
    nb = jupytext.reads(SCRIPT, fmt="py:percent")
    stream = nbformat.v4.new_output
    nb.cells[1].outputs = [
        stream("stream", text="header\n"),
        stream("stream", name="stderr", text="warning\n"),
        stream("stream", text="row 1\n"),
        stream("stream", text="row 2\n"),
        stream("display_data", data={"image/png": PNG}),
        stream("stream", text="after\n"),
    ]
    media = tmp_path / "media"
    media.mkdir()
    page, _ = build_examples.to_markdown(nb, "01_small", "A small example", "", media)
    blocks = re.findall(r"```\{ \.text \.jb-output \}\n(.*?)\n```", page, re.DOTALL)
    assert blocks == ["header\nrow 1\nrow 2", "after"]
    assert "warning" not in page


def test_pages_link_to_their_neighbours_and_the_gallery(tmp_path):
    # The site's nav lists only the gallery, so each page links onwards itself.
    out = tmp_path / "out"
    out.mkdir()
    titles = {"01_a": "First", "02_b": "Second [draft]", "03_c": "Third"}
    entries = [
        build_examples.Entry(name, title, "", None, "cached", 0.0, f"# {title}\n")
        for name, title in titles.items()
    ]
    build_examples.write_pages(entries, settings(tmp_path, out=out))

    def links(name):
        page = (out / f"{name}.md").read_text()
        return re.findall(r"^\[(.*)\]\((.*)\)\{ \.(.*) \}$", page, re.MULTILINE)

    gallery = ("All examples", "index.md", "jb-gallery")
    second = r"Second \[draft\]"
    assert links("01_a") == [gallery, (f"Next: {second}", "02_b.md", "jb-next")]
    assert links("02_b") == [
        ("Previous: First", "01_a.md", "jb-previous"),
        gallery,
        ("Next: Third", "03_c.md", "jb-next"),
    ]
    assert links("03_c") == [(f"Previous: {second}", "02_b.md", "jb-previous"), gallery]
    assert (out / "01_a.md").read_text().startswith("# First\n\n<nav ")


def test_fences_outgrow_backticks_in_the_text():
    assert build_examples.fenced("a ``` b", "text") == "````text\na ``` b\n````"


def test_build_executes_examples_and_writes_the_gallery(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "01_small.py").write_text(SCRIPT)
    # The kernelspec must override an inherited non-interactive backend.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # main() registers its kernelspec through JUPYTER_PATH, and unsets it after.
    monkeypatch.delenv("JUPYTER_PATH", raising=False)
    out = tmp_path / "docs" / "examples"
    code = build_examples.main(
        [
            "--examples",
            str(examples),
            "--out",
            str(out),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--pip-spec",
            "jbubble[examples]==0.2.0",
        ]
    )
    assert code == 0
    assert "JUPYTER_PATH" not in os.environ
    page = (out / "01_small.md").read_text()
    # The figure has at least twice the pixels of its display size, so it
    # stays sharp on high-density screens.
    png = (out / "media" / "01_small_1.png").read_bytes()
    pixels = int.from_bytes(png[16:20], "big")  # the width in the IHDR chunk
    (width,) = re.findall(r'!\[.*\]\(media/01_small_1\.png\)\{ width="(\d+)"', page)
    assert pixels >= 2 * int(width)
    assert (
        "https://colab.research.google.com/github/imperial-nsb/jbubble/blob/"
        "gh-pages/examples/notebooks/01_small.ipynb"
    ) in page
    nb = nbformat.read(out / "notebooks" / "01_small.ipynb", as_version=4)
    assert nb.cells[1].metadata["tags"] == ["install"]
    assert "01_small.md" in (out / "index.md").read_text()
    assert page.endswith("[All examples](index.md){ .jb-gallery }\n</nav>\n")
    # A second run reuses the executed notebook from the cache.
    (cached,) = (tmp_path / "cache").glob("01_small-*.ipynb")
    mtime = cached.stat().st_mtime_ns
    assert (
        build_examples.main(
            [
                "--examples",
                str(examples),
                "--out",
                str(out),
                "--cache-dir",
                str(cached.parent),
            ]
        )
        == 0
    )
    assert cached.stat().st_mtime_ns == mtime


def test_examples_without_a_figure_fail(tmp_path, monkeypatch, capsys):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "01_quiet.py").write_text('# %%\nprint("no figure")\n')
    # main() restores a JUPYTER_PATH that was set before it ran.
    jupyter_path = str(tmp_path / "jupyter")
    monkeypatch.setenv("JUPYTER_PATH", jupyter_path)
    out = tmp_path / "docs" / "examples"
    code = build_examples.main(
        [
            "--examples",
            str(examples),
            "--out",
            str(out),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--pip-spec",
            "jbubble[examples]==0.2.0",
        ]
    )
    assert code == 1
    assert os.environ["JUPYTER_PATH"] == jupyter_path
    assert "01_quiet.py: the example shows no PNG figure." in capsys.readouterr().err
    assert not (out / "01_quiet.md").exists()
    assert not (out / "index.md").exists()


def test_output_directory_must_be_generated(tmp_path):
    out = tmp_path / "docs"
    out.mkdir()
    (out / "index.md").write_text("# Hand-written\n")
    with pytest.raises(SystemExit, match="wasn't written by this tool"):
        build_examples.prepare_output(out)
    assert (out / "index.md").exists()
