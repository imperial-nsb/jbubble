# metrics

Differentiable loss functions for parameter fitting and model comparison.

All functions in this module operate on plain `jax.Array` arguments and are fully differentiable through `jax.grad`. They are designed to be passed as (or called within) the `loss_fn` argument of `fit_parameters`.

```python
from jbubble.metrics import mse_radius, normalised_mse_radius
from jbubble.metrics import mse_emission, normalised_mse_emission
from jbubble.metrics import peak_expansion, peak_expansion_error
```

---

::: jbubble.metrics.mse_radius

::: jbubble.metrics.normalised_mse_radius

::: jbubble.metrics.peak_expansion

::: jbubble.metrics.peak_expansion_error

::: jbubble.metrics.mse_emission

::: jbubble.metrics.normalised_mse_emission

---

## Choosing a loss function

| Loss | Use when |
|---|---|
| `mse_radius` | Direct radius fitting; same units as data |
| `normalised_mse_radius` | Fitting across bubbles of different sizes; scale-invariant |
| `peak_expansion_error` | Only the amplitude matters, not the full waveform shape |
| `mse_emission` | Fitting to hydrophone measurements |
| `normalised_mse_emission` | Fitting to hydrophone data with a known reference level |

Normalised losses are generally preferable for gradient-based fitting because they keep the loss $\mathcal{O}(1)$ regardless of bubble size or driving pressure, making the choice of learning rate more robust.

## Custom loss functions

Any differentiable function of `SimulationResult` fields works as a `loss_fn`. For example, fitting to the power spectral density:

```python
import jax.numpy as jnp

def psd_loss(result, target_psd):
    fft_r = jnp.abs(jnp.fft.rfft(result.radius)) ** 2
    return jnp.mean((fft_r - target_psd) ** 2)

fit_result = fit_parameters(
    ...,
    loss_fn=lambda result: psd_loss(result, measured_psd),
)
```
