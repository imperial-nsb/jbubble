# JAX tips: JIT, vmap, grad, and fitting

jbubble is built on JAX, which means every simulation is JIT-compilable, batchable over parameters, and differentiable. This page covers the key workflows.

---

## JIT compilation

Wrap `run_simulation` (or `solve_eom`) with `jax.jit` to compile the entire solver graph once and execute it efficiently:

```python
import jax
from jbubble import run_simulation, SaveSpec

simulate = jax.jit(run_simulation)

# First call: compilation (slow)
result = simulate(eom, pulse, save_spec=SaveSpec(1000), t_max=10e-6)

# Subsequent calls with the same argument shapes: fast
result2 = simulate(eom2, pulse, save_spec=SaveSpec(1000), t_max=10e-6)
```

!!! warning "Traced and static values"
    `jax.jit` traces every leaf of its arguments, including Python floats, so changing a value doesn't trigger recompilation. Equinox's filtered transformations, such as `eqx.filter_jit` and `eqx.filter_grad`, treat Python floats inside a module as static instead: changing one triggers recompilation, and no gradient flows to it. To differentiate with respect to a value, pass it as a JAX array, or as a `Parameter` to `fit_parameters`.

---

## Batched parameter sweeps with vmap

Use `jax.vmap` to run thousands of simulations simultaneously with different parameters. `GridSweep` automates the Cartesian-product case:

```python
import jax.numpy as jnp
from jbubble.utils.gridsweep import GridSweep
from jbubble import run_simulation, SaveSpec
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.shell import NoShell
from jbubble.bubble.medium import NewtonianMedium
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine


def simulate_bubble(R0, pressure):
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=1e-3),
        R0=R0,
        P_amb=101325,
        rho_L=998,
        c_L=1500,
    )
    pulse = ToneBurst(freq=1e6, pressure=pressure, shape=Sine(), cycle_num=5)
    result = run_simulation(eom, pulse, save_spec=SaveSpec(500), t_max=10e-6)
    return result.radius.max() / R0  # peak expansion ratio


sweep = GridSweep(
    fn=simulate_bubble,
    search_space={
        "R0": jnp.linspace(1e-6, 5e-6, 10),
        "pressure": jnp.array([50e3, 100e3, 200e3, 400e3]),
    },
    batch_size=256,
)

# Full sweep: shape (10, 4) — one scalar output per parameter combination
peak_expansion = sweep.run()
```

`GridSweep` internally applies `jax.vmap` over batches of parameter combinations and handles reshaping back to the grid shape.

For a fully manual sweep over a single parameter axis:

```python
import equinox as eqx

R0_values = jnp.linspace(1e-6, 5e-6, 20)


def make_eom(R0):
    return KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=1e-3),
        R0=R0,
        P_amb=101325,
        rho_L=998,
        c_L=1500,
    )


# Build a batched EoM by stacking along a leading axis
batched_eom = jax.vmap(make_eom)(R0_values)

# vmap run_simulation over the batched EoM
batched_simulate = jax.vmap(
    lambda eom: run_simulation(eom, pulse, save_spec=SaveSpec(500), t_max=10e-6)
)
results = jax.jit(batched_simulate)(batched_eom)
# results.radius has shape (20, 500)
```

---

## Fit parameters to data

[`fit_parameters`][jbubble.fitting.fit_parameters] differentiates a loss through the ODE solve and minimises it with an optax optimiser. Wrap each physical value in [`Parameter`][jbubble.fitting.Parameter], so that one learning rate suits values of any magnitude and bounds hold. In the following example, `make_eom(kappa_s)` builds the equation of motion, `pulse` is the drive, and `measured_radius` is the measured radius trace [m]:

```python
import optax
from jbubble import fit_parameters
from jbubble.fitting import Parameter
from jbubble.metrics import normalised_mse_radius

fit = fit_parameters(
    make_model=lambda p: (make_eom(p["kappa_s"]), pulse),
    params0={"kappa_s": Parameter(1e-9, lower=0.0)},  # initial guess [N s/m]
    loss_fn=lambda result: normalised_mse_radius(result.radius, measured_radius, 2e-6),
    optimizer=optax.adam(0.05),  # about 5 % per step
)
print(fit.params["kappa_s"])
```

To fit several recordings at once, learn a neural constitutive law, handle failed solves, choose solver settings, or estimate uncertainties, see [Fit model parameters to data](fitting.md).

---

## Computing gradients manually

For custom gradient computations (e.g. sensitivity analysis), use `jax.grad` directly:

```python
def peak_expansion(kappa_s):
    eom = make_eom(kappa_s)
    result = run_simulation(eom, pulse, save_spec=SaveSpec(500), t_max=10e-6)
    return result.radius.max() / eom.R0


grad_fn = jax.grad(peak_expansion)
sensitivity = grad_fn(2.4e-9)  # d(peak expansion) / d(kappa_s) at kappa_s = 2.4 nN·s/m
print("Sensitivity:", sensitivity)
```

---

## HDF5 export for large sweeps

Save sweep outputs to HDF5 for post-processing:

```python
from jbubble.utils.io import export_hdf5, load_hdf5
import jax.numpy as jnp

export_hdf5(
    "sweep_results.h5",
    metadata={"description": "R0-pressure sweep", "freq": 1e6},
    R0_values=jnp.linspace(1e-6, 5e-6, 10),
    pressure_values=jnp.array([50e3, 100e3, 200e3, 400e3]),
    peak_expansion=peak_expansion,
)

arrays, meta = load_hdf5("sweep_results.h5")
print(meta["description"])
print(arrays["peak_expansion"].shape)  # (10, 4)
```
