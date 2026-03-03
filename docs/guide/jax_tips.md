# JAX tips

jbubble is built on JAX, which means every simulation is compatible with JAX
transformations: `jit`, `vmap`, and `grad`. This page explains how to use them
effectively.

## x64 mode

JAX defaults to 32-bit floating-point. jbubble **enables 64-bit mode
automatically** when the package is imported:

```python
import jbubble  # sets jax.config.update("jax_enable_x64", True) automatically
```

You do not need to set this manually.

## 1. JIT compilation

Wrapping `run_simulation` (or any jbubble function) with `jax.jit` compiles the
simulation to optimised machine code on the first call. Subsequent calls with
the same argument shapes skip compilation and run significantly faster:

```python
import jax
from jbubble import MarmottantGompertz, Units, run_simulation
from jbubble.pulse import Pulse
from jbubble import shapes
from jbubble.solver import SaveSpec

bubble = MarmottantGompertz(R0=3e-6)
pulse  = Pulse(freq=2e6, pressure=200e3, cycle_num=3, shape=shapes.Sine())
units  = Units()
spec   = SaveSpec(64)

sim = jax.jit(run_simulation)

# First call: compilation + execution (~seconds)
result = sim(bubble, pulse, units=units, save_spec=spec, window_s=20e-6)

# Subsequent calls: execution only (~milliseconds)
result = sim(bubble, pulse, units=units, save_spec=spec, window_s=20e-6)
```

!!! tip
    JIT compilation traces the function with abstract values. Avoid Python
    control flow that depends on JAX array *values* (use `jax.lax.cond` instead
    of `if`). All jbubble internals already follow this rule.

## 2. Batching over parameters with vmap

`jax.vmap` vectorises a function over a batch dimension. This is the preferred
way to run many simulations in parallel — for example, sweeping over bubble
radii:

```python
import jax
import jax.numpy as jnp
from jbubble import MarmottantGompertz, Units, run_simulation
from jbubble.pulse import Pulse
from jbubble import shapes
from jbubble.solver import SaveSpec

def sim_for_r0(r0):
    bubble = MarmottantGompertz(R0=r0)
    pulse  = Pulse(freq=2e6, pressure=200e3, cycle_num=3, shape=shapes.Sine())
    return run_simulation(
        bubble, pulse,
        units=Units(),
        save_spec=SaveSpec(64),
        window_s=20e-6,
    )

r0s     = jnp.linspace(1e-6, 5e-6, 8)   # 8 different radii
results = jax.vmap(sim_for_r0)(r0s)       # batched SimulationResult

print(results.radius.shape)   # (8, 64)
```

You can combine `vmap` with `jit` for maximum performance:

```python
batched_sim = jax.jit(jax.vmap(sim_for_r0))
results = batched_sim(r0s)
```

## 3. Gradient through shell parameters

`jax.grad` can differentiate through the entire simulation with respect to any
leaf parameter. This is useful for fitting shell parameters to measured
radius–time curves:

```python
import jax
from jbubble import MarmottantGompertz, Units, run_simulation
from jbubble.pulse import Pulse
from jbubble import shapes
from jbubble.solver import SaveSpec

def loss(chi):
    bubble = MarmottantGompertz(R0=3e-6, chi=chi)
    pulse  = Pulse(freq=2e6, pressure=200e3, cycle_num=3, shape=shapes.Sine())
    result = run_simulation(
        bubble, pulse,
        units=Units(),
        save_spec=SaveSpec(64),
        window_s=20e-6,
    )
    return ((result.radius - target_radius) ** 2).mean()

grad_loss = jax.grad(loss)
dL_dchi   = grad_loss(chi_init)
```

Gradient computation requires the Gompertz-family models (`MarmottantGompertz`,
`KellerMiksisGompertz`, etc.) rather than `Marmottant`, because the piecewise
σ(R) of `Marmottant` is not differentiable at the transition radii.

## 4. GridSweep for structured parameter sweeps

For structured sweeps over a Cartesian grid of parameters, use `GridSweep` from
`jbubble.utils.gridsweep`. It wraps the vmap pattern and handles result
aggregation automatically. See the [API reference](../api/utils.md) for details.

## 5. Common pitfalls

| Issue | Cause | Fix |
|---|---|---|
| `ConcretizationTypeError` | Python `if` on a JAX array | Replace with `jax.lax.cond` |
| Slow repeated calls | Missing `jax.jit` | Wrap the simulation function with `jit` |
| NaN gradients | Piecewise σ(R) in `Marmottant` | Switch to `MarmottantGompertz` |
| Out-of-memory on vmap | Too large a batch | Reduce batch size or use `jax.lax.map` |
