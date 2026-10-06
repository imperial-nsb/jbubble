# Quickstart

This page takes you from a ready-made bubble to a model that you assemble
yourself, then shows how to plot the result, compile the simulation once
for many runs, and compute the sound that the bubble radiates. Every code
block runs as written, in order.

## Run a preset

A preset returns an equation of motion and a driving pulse with cited,
physically representative defaults. The following code simulates a 2 µm
lipid-coated microbubble, such as a SonoVue bubble, driven by a five-cycle,
1 MHz, 100 kPa tone burst:

```python
from jbubble import run_simulation
from jbubble.utils.presets import lipid_bubble

eom, pulse = lipid_bubble(R0=2e-6, freq=1e6, pressure=100e3)
result = run_simulation(eom, pulse)
print(f"peak R/R0 = {result.radius.max() / eom.R0:.3f}")
```

[`run_simulation`][jbubble.simulation.run_simulation] integrates from
$t = 0$ to [`pulse.t_end`][jbubble.pulse.base.Pulse.t_end], twice the pulse
duration by default, and saves 1024 evenly spaced samples. The three presets
are [`free_bubble`][jbubble.utils.presets.free_bubble],
[`lipid_bubble`][jbubble.utils.presets.lipid_bubble], and
[`thick_shell_bubble`][jbubble.utils.presets.thick_shell_bubble].

## Build the model from parts

An equation of motion combines three physics models:

- A **gas model**: the pressure inside the bubble.
- A **shell model**: surface tension and the stresses of a coating.
- A **medium model**: the viscous and elastic stresses of the surrounding
  liquid or tissue.

The following code builds the model that `lipid_bubble` returns:
[`KellerMiksis`][jbubble.bubble.eom.KellerMiksis] with an SF6 core, a lipid
shell whose surface tension follows the smoothed Marmottant law, and water:

```{.python continuation}
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import LipidShell, SmoothMarmottantSurfaceTension
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine

sigma = SmoothMarmottantSurfaceTension(
    R_buckle_ratio=0.98058,  # buckling radius / R0
    chi=0.5,  # shell elasticity [N/m]
    sigma_rupture=0.072,  # surface tension after rupture [N/m]
)
eom = KellerMiksis(
    gas=PolytropicGas(gamma=1.095),  # SF6
    shell=LipidShell(sigma=sigma, kappa_s=7.5e-9),  # kappa_s [N s/m]
    medium=NewtonianMedium(
        mu=1e-3,  # water viscosity [Pa s]
        rho_L=998.0,  # liquid density [kg/m³]
        c_L=1500.0,  # speed of sound [m/s]
    ),
    R0=2e-6,  # equilibrium radius [m]
    P_amb=101325.0,  # ambient pressure [Pa]
)
pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)

result = run_simulation(eom, pulse)
print(f"peak R/R0 = {result.radius.max() / eom.R0:.3f}")  # same as the preset
```

Every part is an [Equinox](https://docs.kidger.site/equinox/) module, and any
gas works with any shell, medium, and equation of motion. To swap the
physics, change one argument: for example, `NoShell(sigma=0.072)` gives an
uncoated bubble. [Bubble models](bubble_models.md) describes every option.

## Plot the result

[`SimulationResult`][jbubble.simulation.SimulationResult] holds the time
points, the full state, its time derivative, the driving pressure, and a
convergence flag, all in SI units:

```{.python continuation}
import matplotlib.pyplot as plt

plt.style.use("jbubble.style.light")

t_us = result.ts * 1e6  # [µs]
fig, (ax_p, ax_r) = plt.subplots(2, 1, sharex=True, figsize=(7, 4.5))
ax_p.plot(t_us, result.driving_pressure / 1e3, color="#8c959f")
ax_p.set_ylabel("Drive [kPa]")
ax_r.plot(t_us, result.radius / eom.R0, color="C0")
ax_r.set_xlabel("Time [µs]")
ax_r.set_ylabel("$R/R_0$")
plt.show()

print("converged:", bool(result.converged))
```

Besides `result.radius`, the result has `result.radial_velocity`
($\dot R$), `result.radial_acceleration` ($\ddot R$), and the full state
trajectory in `result.state`.

If the solver fails, for example because it reaches its step limit,
`result.converged` is `False` and the samples after the failure are `inf`.
Called outside `jax.jit` and `jax.vmap`, `run_simulation` also warns. To
choose a solver and its settings, see [Solvers and stiffness](solvers.md).

## Compile once, run many times

The first simulation takes about a second, because JAX compiles the solver.
Called directly, `run_simulation` treats the Python floats in a model as
constants, so a new pressure compiles again. Wrapped in `jax.jit`, every
number becomes an input of one compiled program: a new value, such as
another pressure, reuses it and runs in milliseconds. Only a new model
structure, such as another shell class or another `SaveSpec`, compiles
again:

```{.python continuation}
import time

import jax

simulate = jax.jit(run_simulation)
simulate(eom, pulse).radius.block_until_ready()  # compiles

start = time.perf_counter()
for pressure in [50e3, 100e3, 150e3, 200e3]:
    _, pulse_p = lipid_bubble(pressure=pressure)
    peak = simulate(eom, pulse_p).radius.max() / eom.R0
    print(f"{pressure / 1e3:5.0f} kPa: peak R/R0 = {peak:.3f}")
print(f"{(time.perf_counter() - start) * 1e3:.0f} ms for four simulations")
```

To run many simulations at once, batch them with `jax.vmap` or
[`GridSweep`][jbubble.utils.gridsweep.GridSweep]; see
[Parameter sweeps](sweeps.md).

## Compute the radiated pressure

An oscillating bubble radiates sound. The emission models in
`jbubble.acoustics` turn a result into the pressure at a distance `r` from
the bubble.
[`IncompressibleMonopole`][jbubble.acoustics.emission.IncompressibleMonopole]
gives the pressure without the travel time;
[`QuasiAcoustic`][jbubble.acoustics.emission.QuasiAcoustic] gives the same
values on the arrival-time axis `result.ts + r / c_L`. Both take the medium of the equation of motion, so the radiated pressure uses the same liquid density and speed of sound as the simulation:

```{.python continuation}
from jbubble.acoustics import QuasiAcoustic

emission = QuasiAcoustic(medium=eom.medium)  # radiate into the same water
r = 10e-3  # 10 mm from the bubble centre [m]
p_rad = emission(result, r)  # [Pa]
t_arrival = emission.observer_time(result, r)  # [s]
print(f"peak radiated pressure at 10 mm: {abs(p_rad).max():.1f} Pa")
print(f"first sample arrives at {t_arrival[0] * 1e6:.2f} µs")
```

Before you compute a spectrum or resolve an inertial collapse, read the
sampling notes in [`EmissionModel`][jbubble.acoustics.emission.EmissionModel]:
a collapse peak can be shorter than a nanosecond.

## Next steps

- [Bubble models](bubble_models.md): every equation of motion, gas, shell,
  and medium, and how to add your own.
- [Pulse shapes](pulse_shapes.md): tone bursts, chirps, measured waveforms,
  and pulse algebra.
- [Parameter sweeps](sweeps.md) and [Fit model parameters to data](fitting.md):
  batches, gradients, and optimisation.
- [Examples](../examples/index.md): runnable scripts with figures, starting
  with [Your first simulation](../examples/01_first_simulation.md).
