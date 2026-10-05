# AGENTS.md

Instructions for AI coding agents working on jbubble.

## Environment

Use uv. `uv sync` creates `.venv/` from `uv.lock` with Python 3.13, the `io`
and `examples` extras, and the `dev` dependency group. Prefix every Python
command, test run, and example with `uv run`:

```bash
uv sync
MPLBACKEND=Agg uv run python examples/01_first_simulation.py
```

Set `MPLBACKEND=Agg` when you run an example. Every example calls `plt.show()`,
which otherwise opens a window and blocks until someone closes it.

Don't use the old `bubbles` conda environment: it runs Python 3.11, and jbubble
needs Python 3.12 or later.

After you change dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock` with the change.

## Coding conventions

- **JAX-only numerics:** use `jnp` (not `np`) in model code. Keep everything
  JAX-traceable.
- **No host callbacks in traced code:** never call `jax.debug.callback`,
  `jax.pure_callback`, or `jax.experimental.io_callback` in library code that
  runs under tracing. A callback that runs JAX operations can deadlock batched
  runs. To validate parameters, check concrete values at construction in plain
  Python or NumPy, and skip traced values. Tests assert that the jaxpr of
  `run_simulation` contains no `debug_callback`.
- **Property fields:** all `Property` fields use `eqx.field(converter=as_property)` —
  accepts plain `float` or a `Property` instance.
- **EoM return type:** `__call__` always returns
  `BubbleState(R=R_dot, R_dot=R_ddot)`. Omitted fields (R0, P_gas0) default to
  zero derivative.
- **Autodiff for derivatives:** use `jax.grad` inside EoMs; never hand-code
  analytical derivatives.
- **Field ordering:** fields with defaults must follow fields without defaults
  (standard dataclass rule).
- **Docstrings:** Markdown-flavoured numpy style. Document `eqx.Module`
  fields in a `Parameters` section, with units in square brackets. Write
  the governing equation as `$$` LaTeX display math (`$...$` inline) in a
  raw docstring, `r"""..."""`; ruff rule D301 flags a docstring that
  contains a backslash and isn't raw. Link jbubble objects with autorefs,
  such as ``[`KellerMiksis`][jbubble.bubble.eom.KellerMiksis]``; write
  third-party objects in plain backticks. Use fenced ```` ```python ````
  blocks for code samples, not reST `::` blocks or roles. Cite the source of
  every physical default value, such as a preset parameter.
- **Public names:** list each module's public names in its `__all__`, and add
  a `::: dotted.path` directive for each one to the matching page in
  `docs/api/`. `tests/test_docs_api_coverage.py` fails when one is missing.
- **No barrel re-exports:** subpackage classes are NOT re-exported from
  `jbubble/__init__.py`. Users import from their subpackage
  (`jbubble.bubble.eom`, `jbubble.pulse`, etc.). Only top-level orchestration
  functions (`run_simulation`, `fit_parameters`, `SaveSpec`, etc.) live at the
  package root.

## Examples

Each file in `examples/` is a self-contained
[jupytext](https://jupytext.readthedocs.io/) percent-format script that also
runs as a plain Python script. `scripts/build_examples.py` turns each one into
a gallery page and a Colab notebook.

- Start each code cell with `# %%` and each narrative cell with
  `# %% [markdown]`. Create, draw, and show each figure in one cell, ending
  with `plt.show()`.
- Call `plt.style.use("jbubble.style.light")` after you import Matplotlib.
- Colour by role: `C0` for the bubble, `C1` to `C3` for comparisons, and the
  driving pulse in grey on its own axes. Plot at most four categorical series
  on one set of axes.
- Keep the default runtime under about 60 s on a laptop CPU. When an example
  does heavy work, read the `JBUBBLE_QUICK` environment variable, and shrink
  the work when it's `1`, for quick checks.
- Print a short summary of the key numbers.
- Don't hard-code output paths or save files, unless the example is about file
  input and output; then write to a temporary directory.

## Docs

```bash
# Install the docs tools
uv sync --group docs

# Execute the examples and write the gallery and notebooks to docs/examples/
uv run python scripts/build_examples.py

# Build the site in strict mode
uv run zensical build --clean --strict

# Run the code blocks in the README and the guides
uv run pytest --markdown-docs --markdown-docs-syntax=superfences README.md docs/guide
```

`scripts/make_readme_assets.py` generates the README figures in
`docs/assets/readme/`. The README links to them by absolute URL on `main`, so
never rename a file that a released README references.

## Changelog and commits

Add each user-visible change to the top, unreleased section of `CHANGELOG.md`,
in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. List
each breaking change under "Upgrading" with what users need to change. Write
commit messages as [Conventional Commits](https://www.conventionalcommits.org/),
such as `fix(pulse): ...` or `feat(solver)!: ...`.

## Testing

```bash
# Fast suite (excludes slow fitting/integration tests)
uv run pytest -m "not slow"

# Full suite
uv run pytest
```

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check jbubble

# All pre-commit hooks, as CI runs them
uv run prek run --all-files
```

## Key files

| Path | Purpose |
|------|---------|
| `jbubble/bubble/eom.py` | Equations of motion (ODE right-hand sides) |
| `jbubble/bubble/state.py` | BubbleState |
| `jbubble/bubble/property.py` | Property abstraction (`state → scalar`) |
| `jbubble/bubble/gas.py` | Gas models |
| `jbubble/bubble/shell.py` | Shell models and surface tension properties |
| `jbubble/bubble/medium.py` | Medium models |
| `jbubble/pulse/` | Acoustic driving signals |
| `jbubble/simulation.py` | `run_simulation()` and `SimulationResult` |
| `jbubble/solver.py` | `solve_eom()`, `SolverConfig`, and `SaveSpec` |
| `jbubble/fitting.py` | `fit_parameters()`, `Parameter`, and `FitResult` |
| `jbubble/acoustics/emission.py` | Acoustic emission models |
| `jbubble/utils/gridsweep.py` | `GridSweep` parallel parameter sweeps |
| `jbubble/utils/presets.py` | Preset bubble configurations |
| `jbubble/style/` | Matplotlib styles for jbubble figures |
| `examples/` | Example gallery sources (jupytext percent format) |
| `scripts/build_examples.py` | Builds the gallery pages and Colab notebooks |
| `scripts/make_readme_assets.py` | Generates the README figures |
| `docs/` | Documentation site sources (Zensical, `mkdocs.yml`) |
| `CHANGELOG.md` | Release notes |
