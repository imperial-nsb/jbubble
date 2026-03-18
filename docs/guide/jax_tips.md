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

!!! warning "Static vs dynamic"
    JAX traces through `eqx.Module` fields that are `jax.Array`. Fields that are plain Python scalars (`int`, `float`, `bool`) or non-array objects are treated as static and trigger recompilation if changed. Keep parameters as `jnp.array` or leave them as Equinox fields.

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
        R0=R0, P_amb=101325, rho_L=998, c_L=1500,
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
        R0=R0, P_amb=101325, rho_L=998, c_L=1500,
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

## Gradient-based parameter fitting

`fit_parameters` runs gradient descent to minimise a differentiable loss function over the simulation result. It uses `jax.grad` through the entire ODE solve (via diffrax's backpropagation-through-time adjoint).

### Minimal example

```python
import optax
from jbubble import run_simulation, fit_parameters, SaveSpec
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.shell import LipidShell, GompertzSurfaceTension
from jbubble.bubble.medium import NewtonianMedium
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.metrics import normalised_mse_radius

# Measured radius trace (from experiment or higher-fidelity simulation)
measured_radius = ...  # shape (N,) [m]
R0_true = 2e-6

pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)

def make_model(kappa_s):
    sigma = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=LipidShell(sigma=sigma, kappa_s=kappa_s),
        medium=NewtonianMedium(mu=1e-3),
        R0=R0_true, P_amb=101325, rho_L=998, c_L=1500,
    )
    return eom, pulse

fit_result = fit_parameters(
    make_model=make_model,
    params0=1e-9,          # initial guess for kappa_s [N·s/m]
    save_spec=SaveSpec(500),
    t_max=10e-6,
    loss_fn=lambda result: normalised_mse_radius(result.radius, measured_radius, R0_true),
    optimizer=optax.adam(1e-10),
    n_steps=200,
    log_every=25,
)

print("Fitted kappa_s:", fit_result.params)
print("Final loss:", fit_result.loss_history[-1])
```

### Fitting multiple parameters jointly

`params0` can be any JAX-compatible PyTree. The `make_model` factory receives the same structure:

```python
import jax.numpy as jnp

params0 = {"kappa_s": 1e-9, "chi": 0.5}

def make_model(params):
    sigma = GompertzSurfaceTension(
        R_buckle_ratio=0.98,
        chi=params["chi"],
        sigma_rupture=0.072,
    )
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=LipidShell(sigma=sigma, kappa_s=params["kappa_s"]),
        medium=NewtonianMedium(mu=1e-3),
        R0=2e-6, P_amb=101325, rho_L=998, c_L=1500,
    )
    return eom, pulse

fit_result = fit_parameters(
    make_model=make_model,
    params0=params0,
    ...
)
```

### Fitting to emission data

The `loss_fn` receives a full `SimulationResult`, so you can compute any differentiable quantity — including the radiated pressure:

```python
from jbubble.acoustics import IncompressibleMonopole
from jbubble.metrics import normalised_mse_emission

emission = IncompressibleMonopole(rho_L=998.0)
measured_p = ...    # measured hydrophone waveform [Pa]
r = 10e-3           # hydrophone distance [m]

fit_result = fit_parameters(
    make_model=make_model,
    params0=params0,
    save_spec=SaveSpec(500),
    t_max=10e-6,
    loss_fn=lambda result: normalised_mse_emission(
        emission(result, r), measured_p, p_ref=1e3
    ),
    optimizer=optax.adam(1e-10),
    n_steps=300,
)
```

### Monitoring convergence

Pass a `step_callback` to inspect parameters and loss outside JIT at each step:

```python
history = []

def callback(step, params, loss):
    history.append({"step": step, "kappa_s": float(params), "loss": float(loss)})

fit_result = fit_parameters(..., step_callback=callback)
```

---

## Computing gradients manually

For custom gradient computations (e.g. sensitivity analysis), use `jax.grad` directly:

```python
def peak_expansion(kappa_s):
    _, pulse = make_model(kappa_s)
    eom, _ = make_model(kappa_s)
    result = run_simulation(eom, pulse, save_spec=SaveSpec(500), t_max=10e-6)
    return result.radius.max() / eom.R0

grad_fn = jax.grad(peak_expansion)
sensitivity = grad_fn(2.4e-9)   # d(peak expansion) / d(kappa_s) at kappa_s = 2.4 nN·s/m
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
