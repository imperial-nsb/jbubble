# jbubble

**jbubble** is a JAX-based microbubble dynamics library for transcranial focused ultrasound (tFUS) simulation. It provides differentiable ODE models for encapsulated microbubble oscillation, a library of ultrasound pulse shapes, and high-level simulation utilities — all built on JAX for GPU acceleration and automatic differentiation.

## Features

- **Compositional bubble models** — mix and match gas laws, shell coatings, surrounding-medium rheologies, and equations of motion independently
- **5 equations of motion**: Rayleigh–Plesset, Modified Rayleigh–Plesset, Keller–Miksis, Leighton tube confinement, and spherical vessel confinement
- **25+ pulse shapes** implemented as differentiable JAX modules
- **Non-dimensionalisation** via a `Units` class tuned for microbubble length and time scales
- **JIT, vmap, and grad** compatible throughout — enabling fast parameter sweeps and gradient-based optimisation
- **Automatic differentiation through `p_L`** — EoMs that require $\dot{p}_L$ (e.g. Keller–Miksis) compute it via `jax.grad`, so any component combination works out of the box

## Quick start

```python
from jbubble import (
    PolytropicGas, GompertzSigma, LipidShell, KelvinVoigtMedium,
    KellerMiksis, Units, run_simulation,
)
from jbubble.pulse import Pulse
from jbubble import shapes
from jbubble.solver import SaveSpec

R0 = 3e-6
sigma = GompertzSigma.from_R0(R0=R0)
shell  = LipidShell(sigma=sigma)
gas    = PolytropicGas.from_equilibrium(R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0)
medium = KelvinVoigtMedium(R0=R0)
eom    = KellerMiksis(gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=998.0, c_L=1481.0)

pulse  = Pulse(freq=2e6, pressure=200e3, cycle_num=3, shape=shapes.Sine())
result = run_simulation(eom, pulse, units=Units(), save_spec=SaveSpec(64), window_s=20e-6)

print(result.radius)   # shape (64,) — radii at saved time-steps
print(result.ts)       # shape (64,) — saved times in seconds
print(result.converged)
```

## Next steps

- [Installation](guide/installation.md) — how to install jbubble and its dependencies
- [Quickstart](guide/quickstart.md) — a more detailed walk-through of the simulation workflow
- [Bubble models](guide/bubble_models.md) — understand the compositional architecture and choose the right components
- [API reference](api/index.md) — full autodoc reference for every public class and function
