# Contributing to jbubble

Thank you for your interest in contributing to jbubble. This guide explains how to set up a development environment, run the checks, and submit a pull request.

## Set up a development environment

jbubble uses [uv](https://docs.astral.sh/uv/) to manage its development environment. To install uv, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

1. Fork the repository on [GitHub](https://github.com/imperial-nsb/jbubble).

2. Clone your fork, replacing `<your-username>` with your GitHub username:

   ```bash
   git clone https://github.com/<your-username>/jbubble.git
   cd jbubble
   ```

3. Create the environment:

   ```bash
   uv sync
   ```

   This command creates `.venv/` with the Python version in `.python-version` (3.13). It installs jbubble in editable mode with the `examples` extra, plus the `dev` dependency group: pytest, ruff, ty, and prek. All versions come from `uv.lock`.

   To use another supported Python version, set `UV_PYTHON` for your shell session before you run any `uv` command, and then create the environment:

   ```bash
   export UV_PYTHON=3.14
   uv sync
   ```

   Don't rely on `uv sync --python 3.14` alone: the next `uv run` reads `.python-version` and recreates `.venv/` with Python 3.13.

4. Install the Git hooks, which run ruff and other checks on each commit:

   ```bash
   uv run prek install
   ```

5. Create a branch for your changes:

   ```bash
   git checkout -b my-feature
   ```

### Use conda instead of uv

If your workflow depends on conda, create an environment, install jbubble with pip, and install the Git hooks. The `--group` option needs pip 25.1 or later.

```bash
conda create -n jbubble python=3.13 pip
conda activate jbubble
pip install -e ".[examples]" --group dev
prek install
```

pip doesn't read `uv.lock`, so it installs the newest versions that `pyproject.toml` allows. To run the checks in the next section, leave out the `uv run` prefix.

## Development workflow

### Run the checks

Before you open a pull request, make sure all checks pass:

```bash
# Linting, formatting, and file checks (the same hooks as on commit)
uv run prek run --all-files

# Type checking
uv run ty check jbubble

# Tests (fast suite)
uv run pytest -m "not slow"

# Full test suite (includes fitting and integration tests)
uv run pytest
```

To run the tests in parallel, add `-n auto`.

On each pull request, CI runs the fast suite on Python 3.13 with the oldest versions that `pyproject.toml` allows, and on Python 3.14 with the newest releases. Every week, CI runs the full suite on Python 3.13 with the oldest versions and with `uv.lock`, and on Python 3.14 with the newest releases on Linux and macOS. To run the weekly jobs on your branch, start the CI workflow manually from the **Actions** tab.

### Build the docs

[Zensical](https://zensical.org/) builds the documentation site from `docs/` and `mkdocs.yml`. The example gallery comes from the scripts in `examples/`.

1. Install the docs tools:

   ```bash
   uv sync --group docs
   ```

2. Run the examples, and write the gallery and the notebooks to `docs/examples/`:

   ```bash
   uv run python scripts/build_examples.py
   ```

   The first run executes every example and takes a few minutes. The script caches the executed notebooks in `build/examples-cache/`, so later runs only re-execute an example when the example, the jbubble source, or `uv.lock` changes. To write the pages without running anything new, add `--no-execute`; pages without a cached notebook then show code but no output.

3. Preview the site at <http://localhost:8000>. The preview rebuilds when you save a file in `docs/`.

   ```bash
   uv run zensical serve
   ```

4. Before you open a pull request, write the gallery as in step 2, and then build the site in strict mode:

   ```bash
   uv run zensical build --clean --strict
   ```

   With `--strict`, the build fails on broken links and cross-references. It still passes in two cases that CI rejects, so check for them yourself:

   - griffe warns about a docstring. Look for lines that start with `griffe:` in the output. Keep `--clean`: a cached build doesn't repeat the warnings.
   - A nav entry names a page that doesn't exist, such as the gallery before you run step 2. The built site then links to the Markdown file. To list those links, run this command:

     ```bash
     grep -rhoE 'href="[^":]*\.md(#[^"]*)?"' site --include='*.html'
     ```

   For a quick check of the pages alone, the gallery from `scripts/build_examples.py --no-execute` is enough.

CI also runs the code blocks in `README.md` and `docs/guide/`. To run them yourself, use this command:

```bash
uv run pytest --markdown-docs --markdown-docs-syntax=superfences README.md docs/guide
```

To skip a block that can't run on its own, open it with ```` ```{.python notest} ````. To run a block in the namespace of the block before it, open it with ```` ```{.python continuation} ````.

To change the colours of jbubble figures, edit the style sheets in `jbubble/style/`, then regenerate the palette cards in `docs/assets/` with `uv run python scripts/make_palette_card.py`. The site theme in `docs/stylesheets/extra.css` repeats the first two colours of each style sheet and both surfaces, and the logo in `docs/assets/images/logo.svg` uses the first light colour, so update them too. `tests/test_docs_theme.py` checks that they match, and that links and the accent colour keep at least 4.5:1 contrast.

`scripts/make_readme_assets.py` generates the animated figures in `docs/assets/readme/` that the README shows; its docstring explains how to run it. The README links to these files by absolute URL on `main`, so keep their names: a released README on PyPI still points to them.

### Write an example

Each example in `examples/` is a [jupytext](https://jupytext.readthedocs.io/) percent-format script that also runs as a plain Python script. The gallery script turns it into a page and a Colab notebook, and rejects a script without cells or an example that shows no figure. To write an example, follow these conventions:

- Start each code cell with `# %%` and each Markdown cell with `# %% [markdown]`. Create, draw, and show each figure within one cell, and end the cell with `plt.show()`.
- Make the example self-contained, and write the narrative so that it reads well without the rest of the docs.
- Load the jbubble style after you import Matplotlib: `plt.style.use("jbubble.style.light")`.
- Colour by role: `C0` for the bubble, `C1` to `C3` for comparisons, and the driving pulse in grey on its own axes. Plot at most four categorical series on one set of axes.
- Keep the default runtime under about 60 seconds on a laptop CPU. If the example does heavy work, shrink it when the `JBUBBLE_QUICK` environment variable is `1`, for quick checks.
- Print a short summary of the key numbers.
- Don't hard-code output paths or save files, unless the example is about file input and output. In that case, write to a temporary directory.

To run an example without opening figure windows, set `MPLBACKEND=Agg`:

```bash
MPLBACKEND=Agg uv run python examples/01_first_simulation.py
```

### Change dependencies

To add or change a dependency, edit `pyproject.toml`, run `uv lock`, and commit `uv.lock` with your change. The `uv-lock` hook fails if `uv.lock` is out of date. The lower bounds in `pyproject.toml` are the oldest versions that CI tests, so raise one only when jbubble needs a newer release.

### Code style

- **Formatter:** [ruff](https://docs.astral.sh/ruff/) with a line length of 88.
- **Markdown:** don't hard-wrap prose. Write each paragraph or list item on one line, in Markdown files and in the Markdown cells of examples. Code blocks, tables, and HTML keep their own line breaks.
- **Imports:** sorted by ruff (isort rules). No barrel re-exports of subpackage classes from `jbubble/__init__.py` — users import from their subpackage directly.
- **Type annotations:** use standard Python types; JAX arrays are `jax.Array`.
- **Docstrings:** use Markdown-flavoured [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html), which the API reference renders with mkdocstrings. Document `eqx.Module` fields in a `Parameters` section, with units in square brackets. Where applicable, write the governing equation as `$$` LaTeX display math (`$...$` inline), and make the docstring raw (`r"""..."""`) so Python keeps the backslashes; ruff rule D301 checks this. Link jbubble objects with cross-references, such as ``[`KellerMiksis`][jbubble.bubble.eom.KellerMiksis]``, and put code samples in fenced ```` ```python ```` blocks. Cite the source of every physical default value.
- **Public names:** list each module's public names in its `__all__`, and add a `::: dotted.path` directive for each one to the matching page in `docs/api/`. `tests/test_docs_api_coverage.py` fails when one is missing.
- **Use `jnp`** (not `np`) in model code — keep everything JAX-traceable.

### Architecture conventions

If you're adding a new model (gas, shell, medium, EoM), follow the existing patterns:

- All `Property` fields use `eqx.field(converter=as_property)` so users can pass plain `float` values.
- Fields with defaults must follow fields without defaults (dataclass ordering).
- EoM `__call__` returns `BubbleState(R=R_dot, R_dot=R_ddot)` — omitted fields default to zero derivative.
- Use `jax.grad` for all derivatives inside EoMs; never hand-code analytical derivatives.
- Don't call `jax.debug.callback` or any other host callback in code that runs under tracing. A callback that runs JAX operations can deadlock batched runs. To validate parameters, check concrete values at construction in plain Python or NumPy, and skip traced values.
- Test new physics against an independent reference, such as an analytic limit or a published value, not only against the code's own output.

## Submitting a pull request

1. Push your branch to your fork.
2. Open a pull request against `main` on [imperial-nsb/jbubble](https://github.com/imperial-nsb/jbubble).
3. Describe what your change does and why. Link to any relevant issues.
4. If your change affects users, add an entry to the top, unreleased section of [`CHANGELOG.md`](CHANGELOG.md). For a breaking change, also say what users need to change.
5. CI runs the linters, the type checker, the tests, and a build check. All checks must pass.

## Release a version

Maintainers release from `main`. In the release commit, where `YYYY-MM-DD` is the release date:

1. In [`CHANGELOG.md`](CHANGELOG.md), replace `Unreleased` in the new version's heading with the date, as in `## [0.2.0] - YYYY-MM-DD`.
2. At the end of `CHANGELOG.md`, change the version's compare link from `...HEAD` to the new tag, as in `v0.1.1...v0.2.0`.
3. In [`CITATION.cff`](CITATION.cff), check `version` and add `date-released: YYYY-MM-DD`.

Then tag the commit, as in `git tag v0.2.0`, and push the tag. The release workflow tests the tag, uploads the package to PyPI, and then publishes the documentation.

## Reporting bugs and requesting features

Open an issue on [GitHub](https://github.com/imperial-nsb/jbubble/issues). For bugs, include a minimal reproducing example and the full traceback. For feature requests, describe the use case and, if possible, the physics or API you have in mind.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
