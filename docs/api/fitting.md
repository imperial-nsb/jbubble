# fitting

Gradient-based parameter optimisation via JAX autodiff and optax.

```python
from jbubble import fit_parameters, FitResult
```

---

::: jbubble.fitting.FitResult

::: jbubble.fitting.fit_parameters

---

## How it works

`fit_parameters` traces through the entire pipeline:

```
params → make_model(params) → (eom, pulse) → run_simulation → SimulationResult → loss_fn → scalar
```

JAX's autodiff (via diffrax's backpropagation-through-time) computes the gradient of the loss with respect to `params`, which is then fed to an optax optimiser.

The function automatically partitions `params` into:
- **Differentiable leaves**: `jax.Array` scalars and arrays — these receive gradients.
- **Static leaves**: Python scalars, strings, non-array objects — these are held fixed.

This means integer hyperparameters (e.g. number of cycles) can live inside `params` without causing errors.

## Memory considerations

For long simulations with many steps, backpropagation through the ODE solve requires storing intermediate solver states. If memory is a concern, use the recursive checkpoint adjoint:

```python
import diffrax

fit_result = fit_parameters(
    ...,
    adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=100),
)
```

This trades computation for memory: it recomputes intermediate states during the backward pass rather than storing all of them.

## Convergence monitoring

The `step_callback` is called **outside JIT** after each gradient step, so it can perform arbitrary Python-side operations (plotting, logging, early stopping):

```python
import matplotlib.pyplot as plt

losses = []
params_history = []

def callback(step, params, loss):
    losses.append(float(loss))
    params_history.append(params)
    if step % 50 == 0:
        plt.clf()
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.pause(0.01)

fit_result = fit_parameters(..., step_callback=callback)
```
