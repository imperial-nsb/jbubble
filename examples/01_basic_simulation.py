"""01 Basic Simulation

The absolute minimal microbubble simulation.

We simulate a 2 micron radius air bubble in water, driven by a
1 MHz acoustic pulse at 50 kPa. This uses the high-level
'run_simulation' API which handles the integration and
returns a 'SimulationResult' object.
"""

import jax
import matplotlib.pyplot as plt
from jbubble import run_simulation
from jbubble.bubble.eom import RayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.solver import SaveSpec

# 1. Define physics components
# jbubble uses a 'composition' approach: you build an Equation of Motion (EoM)
# by picking a gas law, a shell model, and a medium model.
gas = PolytropicGas(gamma=1.4)
shell = NoShell(sigma=0.072)  # Surface tension of water (0.072 N/m)
medium = NewtonianMedium(mu=0.001)  # Viscosity of water (0.001 Pa s)

# 2. Build the Equation of Motion (EoM)
eom = RayleighPlesset(
    gas=gas,
    shell=shell,
    medium=medium,
    R0=2e-6,  # 2 micron equilibrium radius
    P_amb=101325.0,  # 1 atm ambient pressure
    rho_L=998.0,  # Density of water (kg/m^3)
)

# 3. Define the acoustic driving pulse
# Here we use a 1 MHz ToneBurst with a Sine shape, lasting for 5 cycles.
pulse = ToneBurst(
    freq=1e6,
    pressure=50e3,
    shape=Sine(),
    cycle_num=5,
)

# 4. Run the simulation
# We JIT compile 'run_simulation' for maximum performance.
# 'SaveSpec' controls how many points are saved in the result.
result = jax.jit(run_simulation)(
    eom,
    pulse,
    save_spec=SaveSpec(num_samples=1000),
)

# 5. Visualize the radius over time
plt.figure(figsize=(10, 5))
plt.plot(result.ts * 1e6, result.radius * 1e6, lw=2, color="navy")
plt.xlabel("Time (µs)")
plt.ylabel("Radius (µm)")
plt.title("2 µm Air Bubble in Water (1 MHz, 50 kPa)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
