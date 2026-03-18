# jbubble

**Differentiable microbubble dynamics in JAX.**

jbubble is a research library for simulating and fitting acoustic microbubble dynamics. It is built on [JAX](https://github.com/google/jax), [Equinox](https://github.com/patrick-kidger/equinox), and [diffrax](https://github.com/patrick-kidger/diffrax), making every simulation fully differentiable and JIT-compilable.

## Why jbubble?

| Feature | jbubble | APECSS / MATLAB |
|---|---|---|
| JIT compilation | Yes (`jax.jit`) | No |
| Vectorised sweeps | Yes (`jax.vmap`) | Script loops |
| Gradient-based fitting | Yes (`jax.grad`) | Finite differences |
| Composable physics | Yes (mix any gas+shell+medium) | Fixed combinations |
| Neural components | Yes (`NeuralProperty`, `NeuralPulse`) | No |

## Quick example

```python
import jax
from jbubble import run_simulation, SaveSpec
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.shell import NoShell
from jbubble.bubble.medium import NewtonianMedium
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine

eom = KellerMiksis(
    gas=PolytropicGas(gamma=1.4),
    shell=NoShell(sigma=0.072),
    medium=NewtonianMedium(mu=1e-3),
    R0=2e-6, P_amb=101325.0, rho_L=998.0, c_L=1500.0,
)
pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)

result = jax.jit(run_simulation)(
    eom, pulse,
    save_spec=SaveSpec(num_samples=1000),
    t_max=10e-6,
)

print(result.radius.max() / eom.R0)   # peak expansion ratio
```

Or use a preset:

```python
from jbubble.utils.presets import lipid_bubble

preset = lipid_bubble(R0=2e-6, freq=1e6, pressure=100e3)
result = jax.jit(run_simulation)(
    preset.eom, preset.pulse,
    save_spec=SaveSpec(num_samples=1000),
    t_max=10e-6,
)
```

## Architecture at a glance

jbubble is structured around a **three-way ontological distinction**:

- **[Property](api/bubble.md#properties)** — a function `state → scalar`. Any physical parameter that might depend on the bubble state (surface tension, viscosity, …). A plain `float` is automatically promoted.
- **[BubbleState](api/bubble.md#state)** — the ODE integration variables: radius $R$, wall velocity $\dot{R}$, and frozen equilibrium fields $R_0$, $P_{\text{gas},0}$.
- **Static params** — quantities fixed for the lifetime of a simulation (ambient pressure, density, …), stored as plain fields on the equation of motion.

This separation keeps the API composable: **any** gas model works with **any** shell model and **any** medium model, and all combinations work with all equations of motion automatically via autodiff.

## Navigation

- **[Installation](guide/installation.md)** — environment setup.
- **[Quickstart](guide/quickstart.md)** — first simulation in five minutes.
- **[Bubble models](guide/bubble_models.md)** — guide to gas, shell, and medium models.
- **[Pulse shapes](guide/pulse_shapes.md)** — constructing acoustic driving waveforms.
- **[JAX tips](guide/jax_tips.md)** — JIT, vmap, grad, fitting workflows.
- **[API reference](api/index.md)** — full class and function documentation.
