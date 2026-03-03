# Quickstart

This page walks through a complete simulation from scratch.

## Imports

```python
from jbubble import MarmottantGompertz, Units, run_simulation, SimulationResult
from jbubble.pulse import Pulse
from jbubble import shapes
from jbubble.solver import SaveSpec
```

## 1. Create a bubble

All bubble model constructors accept physical SI parameters. Here we create a
`MarmottantGompertz` bubble with a 3 µm equilibrium radius:

```python
bubble = MarmottantGompertz(R0=3e-6)   # R0 in metres
```

See [Bubble models](bubble_models.md) for a description of every available model
and its parameters.

## 2. Define a pulse

A `Pulse` combines an ultrasound frequency, peak negative pressure amplitude,
number of cycles, and a `PulseShape` waveform object:

```python
pulse = Pulse(
    freq=2e6,        # 2 MHz carrier frequency
    pressure=200e3,     # 200 kPa peak negative pressure
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
    bubble,
    pulse,
    units=units,
    save_spec=SaveSpec(64),  # save 64 evenly-spaced snapshots
    window_s=20e-6,          # simulation window: 20 µs
)
```

## 4. Inspect the result

```python
print(result.radius)     # JAX array, shape (64,) — R/R0 at each saved time
print(result.ts)         # JAX array, shape (64,) — times in seconds
print(result.converged)  # bool — whether the ODE solver converged
```

`SimulationResult` also stores the raw state if needed:

```python
print(result.state)   # full ODE state array at saved time-points
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
from jbubble import MarmottantGompertz, Units, run_simulation
from jbubble.pulse import Pulse
from jbubble import shapes
from jbubble.solver import SaveSpec

bubble = MarmottantGompertz(R0=3e-6)
pulse  = Pulse(freq=2e6, pressure=200e3, cycle_num=3, shape=shapes.Sine())
units  = Units()

result = run_simulation(
    bubble, pulse,
    units=units,
    save_spec=SaveSpec(64),
    window_s=20e-6,
)

print("Converged:", result.converged)
print("Max R/R0:", result.radius.max())

p_rad = result.radiated_pressure(d=1e-3)
```
