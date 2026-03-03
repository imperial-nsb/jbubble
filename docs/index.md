# jbubble

**jbubble** is a JAX-based microbubble dynamics library for transcranial focused ultrasound (tFUS) simulation. It provides differentiable ODE models for encapsulated microbubble oscillation, a library of ultrasound pulse shapes, and high-level simulation utilities — all built on JAX for GPU acceleration and automatic differentiation.

## Features

- **7 bubble models** spanning bare gas bubbles through shell-encapsulated agents in viscoelastic or confined media
- **25+ pulse shapes** implemented as differentiable JAX modules
- **Non-dimensionalisation** via a `Units` class tuned for microbubble length and time scales
- **JIT, vmap, and grad** compatible throughout — enabling fast parameter sweeps and gradient-based optimisation
- **Fixed-step ODE solver** returning compact `SimulationResult` arrays suitable for downstream ML pipelines

## Quick start

```python
from jbubble import MarmottantGompertz, Units, run_simulation
from jbubble.pulse import Pulse
from jbubble import shapes
from jbubble.solver import SaveSpec

bubble = MarmottantGompertz(R0=3e-6)
pulse  = Pulse(freq=2e6, pressure=200e3, cycle_num=3, shape=shapes.Sine())
result = run_simulation(bubble, pulse, units=Units(), save_spec=SaveSpec(64), window_s=20e-6)

print(result.radius)   # shape (64,) — normalised radii at saved time-steps
print(result.ts)       # shape (64,) — saved times in seconds
print(result.converged)
```

## Next steps

- [Installation](guide/installation.md) — how to install jbubble and its dependencies
- [Quickstart](guide/quickstart.md) — a more detailed walk-through of the simulation workflow
- [Bubble models](guide/bubble_models.md) — choose the right model for your application
- [API reference](api/index.md) — full autodoc reference for every public class and function
