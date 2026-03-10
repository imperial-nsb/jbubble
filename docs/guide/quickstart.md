# Quickstart

This page walks through a complete simulation from scratch.

## Imports

```python
from jbubble import (
    PolytropicGas, GompertzSigma, LipidShell, KelvinVoigtMedium,
    KellerMiksis, Units, run_simulation, SimulationResult,
)
from jbubble.pulse import Pulse
from jbubble import shapes
from jbubble.solver import SaveSpec
```

## 1. Compose a bubble model

jbubble uses a compositional architecture: you assemble a bubble model from
independent **gas**, **surface tension**, **shell**, and **medium** components,
then plug them into an **equation of motion** (EoM).

```python
R0 = 3e-6  # equilibrium radius [m]

# Surface tension law (smooth Gompertz sigmoid)
sigma = GompertzSigma.from_R0(R0=R0)

# Shell coating (lipid monolayer)
shell = LipidShell(sigma=sigma)

# Gas law (polytropic, initialised at equilibrium)
gas = PolytropicGas.from_equilibrium(
    R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0,
)

# Surrounding medium (Newtonian liquid — default parameters for water)
medium = KelvinVoigtMedium(R0=R0)

# Equation of motion (Keller–Miksis with automatic dp_L/dt via jax.grad)
eom = KellerMiksis(
    gas=gas, shell=shell, medium=medium,
    R0=R0, P_amb=101325.0, rho_L=998.0, c_L=1481.0,
)
```

See [Bubble models](bubble_models.md) for all available components and how to
combine them.

## 2. Define a pulse

A `Pulse` combines an ultrasound frequency, peak negative pressure amplitude,
number of cycles, and a `PulseShape` waveform object:

```python
pulse = Pulse(
    freq=2e6,        # 2 MHz carrier frequency
    pressure=200e3,  # 200 kPa peak negative pressure
    cycle_num=3,     # 3 cycles
    shape=shapes.Sine(),
)
```

See [Pulse shapes](pulse_shapes.md) for the full list of available shapes.

## 3. Run the simulation

`run_simulation` integrates the bubble ODE over a time window and returns a
`SimulationResult`. The `SaveSpec` controls how many time-points are stored:

```python
units  = Units()           # default non-dimensionalisation (L=1µm, T=1µs)
result = run_simulation(
    eom,
    pulse,
    units=units,
    save_spec=SaveSpec(64),  # save 64 evenly-spaced snapshots
    window_s=20e-6,          # simulation window: 20 µs
)
```

## 4. Inspect the result

```python
print(result.radius)     # JAX array, shape (64,) — R(t) at each saved time
print(result.ts)         # JAX array, shape (64,) — times in seconds
print(result.converged)  # bool — whether the ODE solver converged
```

`SimulationResult` also stores the raw state if needed:

```python
print(result.radial_velocity)      # dR/dt
print(result.radial_acceleration)  # d²R/dt²
```

## 5. Compute radiated pressure

The far-field radiated pressure at a distance `d` from the bubble can be
computed from the simulation result:

```python
p_rad = result.radiated_pressure(d=1e-3)  # pressure waveform at 1 mm
```

The returned array has the same length as `result.ts`.

## Complete example

```python
from jbubble import (
    PolytropicGas, GompertzSigma, LipidShell, KelvinVoigtMedium,
    KellerMiksis, Units, run_simulation,
)
from jbubble.pulse import Pulse
from jbubble import shapes
from jbubble.solver import SaveSpec

R0 = 3e-6
sigma  = GompertzSigma.from_R0(R0=R0)
shell  = LipidShell(sigma=sigma)
gas    = PolytropicGas.from_equilibrium(R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0)
medium = KelvinVoigtMedium(R0=R0)
eom    = KellerMiksis(gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=998.0, c_L=1481.0)

pulse  = Pulse(freq=2e6, pressure=200e3, cycle_num=3, shape=shapes.Sine())
units  = Units()

result = run_simulation(
    eom, pulse,
    units=units,
    save_spec=SaveSpec(64),
    window_s=20e-6,
)

print("Converged:", result.converged)
print("Max R/R0:", float(result.radius.max()) / R0)

p_rad = result.radiated_pressure(d=1e-3)
```
