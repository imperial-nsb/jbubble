# Quickstart

This page walks through a complete simulation from scratch: choosing an equation of motion, constructing a driving pulse, running the solver, and inspecting the output.

## 1. Choose an equation of motion

An `EquationOfMotion` (EoM) is the ODE right-hand side. It bundles three physics sub-models:

- **Gas model** — pressure inside the bubble.
- **Shell model** — surface tension and shell stresses.
- **Medium model** — viscous and elastic stresses from the surrounding liquid.

```python
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.shell import NoShell
from jbubble.bubble.medium import NewtonianMedium

eom = KellerMiksis(
    gas=PolytropicGas(gamma=1.4),
    shell=NoShell(sigma=0.072),       # water surface tension [N/m]
    medium=NewtonianMedium(mu=1e-3),  # water viscosity [Pa·s]
    R0=2e-6,       # equilibrium radius [m]
    P_amb=101325,  # ambient pressure [Pa]
    rho_L=998.0,   # liquid density [kg/m³]
    c_L=1500.0,    # speed of sound [m/s]
)
```

`KellerMiksis` is the standard choice: it accounts for first-order liquid compressibility and is well-tested across a wide range of driving pressures.

## 2. Construct a driving pulse

```python
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine

pulse = ToneBurst(
    freq=1e6,       # centre frequency [Hz]
    pressure=100e3, # peak pressure [Pa]
    shape=Sine(),   # sinusoidal carrier
    cycle_num=5,    # number of cycles
)
```

The pulse is a callable `p_ac(t)` returning pressure at time `t`.

## 3. Run the simulation

```python
import jax
from jbubble import run_simulation, SaveSpec

result = jax.jit(run_simulation)(
    eom, pulse,
    save_spec=SaveSpec(num_samples=1000),
    t_max=10e-6,  # simulate 10 µs
)
```

Wrapping with `jax.jit` compiles the entire solver graph. The first call pays a one-time compilation cost; subsequent calls with the same argument shapes are fast.

## 4. Inspect the result

`SimulationResult` provides convenient properties:

```python
import matplotlib.pyplot as plt

# Check the solver converged
assert result.converged, "ODE did not converge — increase max_steps or loosen tolerances"

# Time axis
ts = result.ts * 1e6          # convert to µs

# Radius normalised by R0
R_norm = result.radius / eom.R0

plt.plot(ts, R_norm)
plt.xlabel("Time [µs]")
plt.ylabel("R / R₀")
plt.title("Keller–Miksis: free bubble at 100 kPa, 1 MHz")
plt.show()

print("Peak expansion ratio:", float(R_norm.max()))
print("Minimum radius ratio:", float(R_norm.min()))
```

The result also carries `state_dot` (time derivatives of all state variables) and `driving_pressure`.

## 5. Using presets

If you just want a physically representative bubble without choosing every parameter, use one of the built-in presets:

```python
from jbubble.utils.presets import free_bubble, lipid_bubble, thick_shell_bubble

# Uncoated bubble in water
preset = free_bubble(R0=2e-6, freq=1e6, pressure=100e3)

# Clinical lipid-shelled UCA (SonoVue-like)
preset = lipid_bubble(R0=1.5e-6, freq=2.5e6, pressure=50e3)

# Polymer thick-shelled UCA (Optison-like)
preset = thick_shell_bubble(R0=2e-6, freq=1e6, pressure=150e3)

result = jax.jit(run_simulation)(
    preset.eom, preset.pulse,
    save_spec=SaveSpec(num_samples=1000),
    t_max=10e-6,
)
```

## 6. Acoustic emission

To compute the radiated pressure at a field point:

```python
from jbubble.acoustics import IncompressibleMonopole

emission = IncompressibleMonopole(rho_L=998.0)
r = 1e-2  # 1 cm from the bubble centre

p_rad = emission(result, r)  # shape (num_samples,) [Pa]
```

For retarded-time corrections:

```python
from jbubble.acoustics import QuasiAcoustic

emission = QuasiAcoustic(rho_L=998.0, c_L=1500.0)
p_rad = emission(result, r)
```

## Next steps

- [Bubble models](bubble_models.md) — detailed guide to gas, shell, and medium options.
- [Pulse shapes](pulse_shapes.md) — constructing complex waveforms.
- [JAX tips](jax_tips.md) — batched sweeps, gradient-based fitting.
- [API reference](../api/index.md) — complete class documentation.
