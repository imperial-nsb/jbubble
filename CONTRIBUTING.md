# Contributing to jbubble

Thank you for your interest in contributing to jbubble! This guide will help
you get set up and familiar with our development workflow.

## Set up a development environment

jbubble uses [uv](https://docs.astral.sh/uv/) to manage its development
environment. To install uv, follow the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

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

   This command creates `.venv/` with the Python version in `.python-version`
   (3.13). It installs jbubble in editable mode with the `io` and `examples`
   extras, plus the `dev` dependency group: pytest, ruff, ty, and prek. All
   versions come from `uv.lock`. To use another supported Python version, run
   `uv sync --python 3.12`, for example.

4. Install the Git hooks, which run ruff and other checks on each commit:

   ```bash
   uv run prek install
   ```

5. Create a branch for your changes:

   ```bash
   git checkout -b my-feature
   ```

### Use conda instead of uv

If your workflow depends on conda, create an environment and install jbubble
with pip. The `--group` option needs pip 25.1 or later.

```bash
conda create -n jbubble python=3.13 pip
conda activate jbubble
pip install -e ".[io,examples]" --group dev
```

pip doesn't read `uv.lock`, so it installs the newest versions that
`pyproject.toml` allows. To run the checks in the next section, leave out the
`uv run` prefix.

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

CI also runs the full suite on Python 3.12, 3.13, and 3.14, both with the
newest dependency releases and with the oldest versions that `pyproject.toml`
allows.

### Change dependencies

To add or change a dependency, edit `pyproject.toml`, run `uv lock`, and commit
`uv.lock` with your change. The `uv-lock` hook fails if `uv.lock` is out of
date. The lower bounds in `pyproject.toml` are the oldest versions that CI
tests, so raise one only when jbubble needs a newer release.

### Code style

- **Formatter:** [ruff](https://docs.astral.sh/ruff/) with a line length of 88.
- **Imports:** sorted by ruff (isort rules). No barrel re-exports of subpackage
  classes from `jbubble/__init__.py` — users import from their subpackage directly.
- **Type annotations:** use standard Python types; JAX arrays are `jax.Array`.
- **Docstrings:** include the governing equation in a `::` code block where applicable.
- **Use `jnp`** (not `np`) throughout — keep everything JAX-traceable.

### Architecture conventions

If you're adding a new model (gas, shell, medium, EoM), follow the existing patterns:

- All `Property` fields use `eqx.field(converter=as_property)` so users can pass
  plain `float` values.
- Fields with defaults must follow fields without defaults (dataclass ordering).
- EoM `__call__` returns `BubbleState(R=R_dot, R_dot=R_ddot)` — omitted fields
  default to zero derivative.
- Use `jax.grad` for all derivatives inside EoMs; never hand-code analytical
  derivatives.

## Submitting a pull request

1. Push your branch to your fork.
2. Open a pull request against `main` on [imperial-nsb/jbubble](https://github.com/imperial-nsb/jbubble).
3. Describe what your change does and why. Link to any relevant issues.
4. CI runs the linters, the type checker, the tests, and a build check. All
   checks must pass.

## Reporting bugs and requesting features

Open an issue on [GitHub](https://github.com/imperial-nsb/jbubble/issues). For
bugs, include a minimal reproducing example and the full traceback. For feature
requests, describe the use case and, if possible, the physics or API you have in
mind.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
