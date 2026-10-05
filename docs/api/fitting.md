# fitting

Gradient-based parameter estimation through the ODE solve, with any optax
optimiser. For a task-oriented introduction, see
[Fit model parameters to data](../guide/fitting.md).

```python
from jbubble import FitResult, fit_parameters
from jbubble.fitting import Parameter, unwrap
```

---

::: jbubble.fitting.fit_parameters

::: jbubble.fitting.FitResult

::: jbubble.fitting.Parameter

::: jbubble.fitting.unwrap

---

## How a step works

`fit_parameters` evaluates the loss and its gradient in one compiled call,
with reverse-mode differentiation through the ODE solve. Each step then does
the following:

1. Computes an update with `optimizer.update(grads, opt_state, params)`.
2. Tries the candidate `params + update`. If any solve fails to converge, or
   the loss or gradient isn't finite, it halves the update and tries again,
   up to `max_backtracks` times.
3. Accepts the first candidate that succeeds. Its loss goes into
   `loss_history`, and its gradient drives the next step.

If every halved update fails, the fit warns and returns the last accepted
parameters. Nothing raises from inside a compiled function, so a failed solve
never ends a fit.

## What gets fitted

| Leaf of `params0` | Fitted | Coordinate that the optimiser updates |
|---|---|---|
| `Parameter(x)` | Yes | `x / scale` |
| `Parameter(x, lower=a)` | Yes | `log((x - a) / scale)` |
| `Parameter(x, upper=b)` | Yes | `log((b - x) / scale)` |
| `Parameter(x, lower=a, upper=b)` | Yes | `logit((x - a) / (b - a))` |
| `Parameter(x, fixed=True)` | No | |
| Python float or `np.float64` in a dict, list, or tuple, or as `params0` | Yes | as `Parameter(x)` |
| Floating-point JAX or NumPy array, or another NumPy scalar | Yes | the value itself |
| Python float inside an Equinox module | No, with a warning if you set it | |
| Integer, boolean, string, or callable | No | |

`make_model`, `loss_fn`, `step_callback`, and `FitResult.params` all see
physical values.

## Gradients through the ODE solve

[`solve_eom`][jbubble.solver.solve_eom] evaluates the equation of motion
through a guard, so a trial step that the adaptive solver rejects, for example
one that overshoots a strong collapse to `R <= 0`, can't make the gradient
NaN. With an explicit solver, such as the default `Dopri5`, the guard leaves
the solution unchanged, bit for bit.
