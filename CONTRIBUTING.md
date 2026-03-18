# Contributing to jbubble

Thank you for your interest in contributing to jbubble! This guide will help
you get set up and familiar with our development workflow.

## Getting started

1. Fork the repository on [GitHub](https://github.com/imperial-nsb/jbubble).

2. Clone your fork and install in development mode:

   ```bash
   git clone https://github.com/<your-username>/jbubble.git
   cd jbubble
   pip install -e ".[dev]"
   ```

3. Create a branch for your changes:

   ```bash
   git checkout -b my-feature
   ```

## Development workflow

### Running checks

Before submitting a pull request, make sure all checks pass:

```bash
# Linting and formatting
ruff check .
ruff format --check .

# Type checking
ty check jbubble

# Tests (fast suite)
pytest tests/ -m "not slow"

# Full test suite (includes fitting and integration tests)
pytest tests/
```

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
4. CI will run lint, type checking, and tests automatically. All checks must pass.

## Reporting bugs and requesting features

Open an issue on [GitHub](https://github.com/imperial-nsb/jbubble/issues). For
bugs, include a minimal reproducing example and the full traceback. For feature
requests, describe the use case and, if possible, the physics or API you have in
mind.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
