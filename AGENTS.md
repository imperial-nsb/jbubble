# AGENTS.md

Instructions for AI coding agents working on jbubble.

## Environment

Use the `bubbles` conda environment:

```bash
conda activate bubbles
```

Run this before any Python commands, tests, or examples.

## Coding conventions

- **JAX-only numerics:** use `jnp` (not `np`) throughout. Keep everything
  JAX-traceable.
- **Property fields:** all `Property` fields use `eqx.field(converter=as_property)` —
  accepts plain `float` or a `Property` instance.
- **EoM return type:** `__call__` always returns
  `BubbleState(R=R_dot, R_dot=R_ddot)`. Omitted fields (R0, P_gas0) default to
  zero derivative.
- **Autodiff for derivatives:** use `jax.grad` inside EoMs; never hand-code
  analytical derivatives.
- **Field ordering:** fields with defaults must follow fields without defaults
  (standard dataclass rule).
- **Docstrings:** include the governing equation in a `::` code block.
- **No barrel re-exports:** subpackage classes are NOT re-exported from
  `jbubble/__init__.py`. Users import from their subpackage
  (`jbubble.bubble.eom`, `jbubble.pulse`, etc.). Only top-level orchestration
  functions (`run_simulation`, `fit_parameters`, `SaveSpec`, etc.) live at the
  package root.

## Testing

```bash
# Fast suite (excludes slow fitting/integration tests)
pytest tests/ -m "not slow"

# Full suite
pytest tests/
```

## Linting and formatting

```bash
ruff check .
ruff format --check .
ty check jbubble
```

## Key files

| Path | Purpose |
|------|---------|
| `planning/context.md` | Authoritative project context — read first, update after changes |
| `jbubble/bubble/eom.py` | Equations of motion (ODE right-hand sides) |
| `jbubble/bubble/state.py` | BubbleState and ConfinedBubbleState |
| `jbubble/bubble/property.py` | Property abstraction (`state → scalar`) |
| `jbubble/bubble/gas.py` | Gas models |
| `jbubble/bubble/shell.py` | Shell models and surface tension properties |
| `jbubble/bubble/medium.py` | Medium models |
| `jbubble/pulse/` | Acoustic driving signals |
| `jbubble/simulation.py` | `run_simulation()` and `SimulationResult` |
| `jbubble/fitting.py` | `fit_parameters()` gradient-based optimisation |
| `jbubble/acoustics/emission.py` | Acoustic emission models |
| `jbubble/utils/presets.py` | Preset bubble configurations |
