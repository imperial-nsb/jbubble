# simulation

High-level simulation runner and result container.

```python
from jbubble import run_simulation, SaveSpec
```

---

::: jbubble.simulation.SimulationResult

::: jbubble.simulation.run_simulation

---

## Working with SimulationResult

`SimulationResult` is an Equinox module, so it is a JAX PyTree. All fields are `jax.Array` and can be passed through `jax.jit`, `jax.vmap`, and `jax.grad`.

```python
result = jax.jit(run_simulation)(eom, pulse, save_spec=SaveSpec(1000), t_max=10e-6)

# Convenience properties
ts  = result.ts                  # shape (N,) — time axis [s]
R   = result.radius              # shape (N,) — bubble radius [m]
Rd  = result.radial_velocity     # shape (N,) — wall velocity [m/s]
Rdd = result.radial_acceleration # shape (N,) — wall acceleration [m/s²]
pac = result.driving_pressure    # shape (N,) — acoustic pressure [Pa]

# Full state and derivatives
state     = result.state         # BubbleState with shape-(N,) arrays
state_dot = result.state_dot     # BubbleState time derivatives

# Solver diagnostics
ok = result.converged            # bool — False if ODE hit max_steps
```

### Confined bubble (SphericalConfinement EoM)

When the EoM is `SphericalConfinement`, the result state is a `ConfinedBubbleState`:

```python
from jbubble.bubble.eom import SphericalConfinement

eom = SphericalConfinement(...)
result = jax.jit(run_simulation)(eom, pulse, save_spec=SaveSpec(1000), t_max=10e-6)

print(result.has_vessel)       # True
a   = result.vessel_radius     # shape (N,) — vessel wall radius [m]
a_d = result.vessel_velocity   # shape (N,) — vessel wall velocity [m/s]
```
