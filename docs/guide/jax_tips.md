# JAX tips

jbubble models are [Equinox](https://docs.kidger.site/equinox/) modules, and a simulation is a pure function of them. This page collects what you need to know to compile, batch, differentiate, and debug simulations with JAX.

Every code block on this page runs as written, in order.

## Models are PyTrees

An equation of motion, with its gas, shell, and medium, is a PyTree: a nested structure whose leaves are the numbers of the model. JAX transformations act on those leaves. Modules are immutable, so to change a value, build a new model, or use `eqx.tree_at`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from jbubble import run_simulation
from jbubble.utils.presets import lipid_bubble

eom, pulse = lipid_bubble()
stiffer = eqx.tree_at(lambda m: m.shell.kappa_s.val, eom, 1.5e-8)
print(eom.shell.kappa_s, "->", stiffer.shell.kappa_s)
```

Coefficients such as `kappa_s` are [`Property`][jbubble.bubble.property.Property] objects; a constant one keeps its number in `val`. `eqx.tree_at` bypasses the constructor, so pass the leaf, as here, or a whole `Property`.

## Precision

Importing jbubble turns on JAX's 64-bit mode for the whole process, because the radius and the wall velocity of a collapsing bubble span many orders of magnitude. Arrays that you create with `jax.numpy` afterwards are `float64` by default. Don't turn the mode off: single precision isn't accurate enough for the solver tolerances.

## Compile with `jax.jit`

`jax.jit(run_simulation)` compiles the whole simulation, solver and post-processing, into one program. JAX compiles again only when the structure of the arguments changes:

| Change | Compiles again? |
|---|---|
| A number in the model or pulse, such as `R0`, `kappa_s`, or `pressure` | No |
| The class of a part, such as `NoShell` instead of `LipidShell` | Yes |
| `SaveSpec(num_samples=...)` or `SolverConfig(max_steps=...)` | Yes: these are static |
| The solver or the step-size controller class | Yes |

Without `jax.jit`, `run_simulation` still compiles the solver, but it treats Python floats in the model as constants, so a new value compiles again. To time a compiled function, call it once to compile, and call `block_until_ready()` on the output, because JAX dispatches work asynchronously.

Equinox's `eqx.filter_jit` and `eqx.filter_grad` treat a Python float inside a module as static: a new value compiles again, and no gradient flows to it. To differentiate a model with respect to all of its numbers, convert them to arrays first, as in the next section.

## Differentiate a whole model

`jax.grad` differentiates with respect to a float, an array, or any PyTree of them. To get the sensitivity of a result to every parameter of a model at once, convert the model's numbers to arrays and use `eqx.filter_grad`, which returns a model-shaped PyTree of derivatives. The following code computes the relative sensitivity, $\partial \ln y / \partial \ln p$, of the peak radius to each parameter of the lipid preset. Gradients through a lipid shell need tighter tolerances than the default; see [Gradients and adjoints](solvers.md#gradients-and-adjoints):

```{.python continuation}
import diffrax

from jbubble import SolverConfig

precise = SolverConfig(stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-12))
model = jax.tree.map(jnp.asarray, eom)  # Python floats -> arrays


def peak_ratio(m):
    return run_simulation(m, pulse, config=precise).radius.max() / m.R0


value, grads = eqx.filter_value_and_grad(peak_ratio)(model)
for name, get in [
    ("R0", lambda m: m.R0),
    ("chi", lambda m: m.shell.sigma.chi.val),
    ("kappa_s", lambda m: m.shell.kappa_s.val),
    ("gamma", lambda m: m.gas.gamma.val),
    ("mu", lambda m: m.medium.mu.val),
]:
    print(f"{name:8s} {get(grads) * get(model) / value:+.4f}")
```

To compute such sensitivities for a batch of bubbles, wrap the gradient in `jax.vmap`; the example [Gradients and optimisation](../examples/09_gradients_and_optimisation.md) shows how, with a finite-difference check.

## Batch with `jax.vmap`

`jax.vmap` maps a function over a leading axis of its inputs. The simplest pattern builds the model inside the function from the swept values, as [Parameter sweeps](sweeps.md) shows. You can also map over a batch of models: `jax.vmap(make_model)(values)` returns one model whose leaves carry the batch axis, and `jax.vmap(simulate)` accepts it directly:

```{.python continuation}
def simulate(m):
    return run_simulation(m, pulse).radius.max() / m.R0


kappas = jnp.array([2.5e-9, 5e-9, 7.5e-9, 1e-8])  # [N s/m]
batch = jax.vmap(lambda k: eqx.tree_at(lambda m: m.shell.kappa_s.val, model, k))(kappas)
print(jax.jit(jax.vmap(simulate))(batch))
```

For grids of parameters, use [`GridSweep`][jbubble.utils.gridsweep.GridSweep], which also runs chunks of the grid in parallel.

## Debug a simulation

- **The solve failed.** Check `result.converged`. Under `jax.jit` and `jax.vmap`, `run_simulation` can't warn; see [Check that a solve converged](solvers.md#check-that-a-solve-converged).
- **A gradient is NaN or very large.** Check that the solve converged, then check whether the problem is stiff; see [When stiffness matters](solvers.md#when-stiffness-matters).
- **You want to see a traced value.** Use `jax.debug.print("{x}", x=x)` inside a compiled function; `print` shows only the tracer.
- **You want to find the first NaN.** Run `jax.config.update("jax_debug_nans", True)` before the simulation; JAX then raises at the operation that produced it.

## Fit parameters to data

[`fit_parameters`][jbubble.fitting.fit_parameters] combines these pieces: it compiles the simulation, differentiates a loss through it, and steps an [optax](https://optax.readthedocs.io) optimiser, with bounds, several recordings at once, and retries for failed solves. See [Fit model parameters to data](fitting.md).
