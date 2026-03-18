# acoustics

The `jbubble.acoustics` module provides models for computing the acoustic pressure radiated by a bubble, given a solved trajectory.

Emission models are **not** `Property` subclasses (they depend on the full trajectory, not just instantaneous state), and **not** part of the EoM (they are pure post-processing). They take a `SimulationResult` and a field-point distance `r` and return the pressure time series.

```python
from jbubble.acoustics import IncompressibleMonopole

emission = IncompressibleMonopole(rho_L=998.0)
p_rad = emission(result, r=1e-2)   # pressure [Pa] at 1 cm, shape (num_samples,)
```

For multiple field-point distances, use `jax.vmap`:

```python
import jax
import jax.numpy as jnp

distances = jnp.array([1e-3, 5e-3, 1e-2, 5e-2])
p_all = jax.vmap(lambda r: emission(result, r))(distances)
# shape: (4, num_samples)
```

---

::: jbubble.acoustics.emission.EmissionModel

::: jbubble.acoustics.emission.IncompressibleMonopole

::: jbubble.acoustics.emission.QuasiAcoustic

---

## Choosing an emission model

| Model | Accuracy | Validity | Speed |
|---|---|---|---|
| `IncompressibleMonopole` | Low | $r \gg R$, $\text{Ma} \ll 1$ | Fastest — instant post-processing |
| `QuasiAcoustic` | Medium | $r \gg R$, moderate Ma | Fast — uses `jnp.interp` |

The fully compressible acoustic emission model (analogous to APECSS's wave equation integration) is planned but not yet implemented. For most research purposes, `QuasiAcoustic` at a field point distance $r \gg R_0$ is adequate.
