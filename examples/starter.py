"""jbubble starter — minimal end-to-end simulation.

Demonstrates the core API:

  1. Build an equation of motion from composable components
  2. Drive it with an acoustic pulse
  3. Run the simulation and inspect results

Run with:
    python examples/starter.py
"""

import jax
import jax.numpy as jnp

from jbubble import run_simulation
from jbubble.bubble.eom import KellerMiksis, RayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium, KelvinVoigtMedium
from jbubble.bubble.properties import GompertzSurfaceTension
from jbubble.bubble.shell import LipidShell
from jbubble.presets import WATER_MU
from jbubble.pulse import Pulse
from jbubble.shapes import Sine
from jbubble.solver import SaveSpec

import matplotlib.pyplot as plt

pressure = 300e3  # 200 kPa

sigma = GompertzSurfaceTension(
    R0=2e-6,
    R_buckle=1.98e-6,
    chi=0.38,
    sigma_break=72e-3,
)

shell  = LipidShell(sigma=sigma, kappa_s=2.4e-9)
gas    = PolytropicGas(gamma=1.07)
mediumA = NewtonianMedium(mu=WATER_MU)
mediumB = KelvinVoigtMedium(mu=WATER_MU, G=1e6, R0=2e-6)

eomA = RayleighPlesset(
    gas=gas,
    shell=shell,
    medium=mediumA,
    R0=2e-6,
    P_amb=101325.0,
    rho_L=998.0,
    # c_L=1481.0,
)

eomB = RayleighPlesset(
    gas=gas,
    shell=shell,
    medium=mediumB,
    R0=2e-6,
    P_amb=101325.0,
    rho_L=998.0,
)

# ---------------------------------------------------------------------------
# 2.  Define the driving pulse
# ---------------------------------------------------------------------------

pulse = Pulse(
    freq=1e6,        # 1 MHz carrier
    pressure=pressure,  # 200 kPa peak
    shape=Sine(),
    cycle_num=10.0,
    apply_hann=True,
)

resultA = jax.jit(run_simulation)(
    eomA,
    pulse,
    save_spec=SaveSpec(num_samples=1024),
    window_s=20e-6,
)

print(f"Converged : {bool(resultA.converged)}")
print(f"R0        : {eomA.R0 * 1e6:.2f} µm")
print(f"R_max     : {float(jnp.max(resultA.radius)) * 1e6:.2f} µm")
print(f"R_min     : {float(jnp.min(resultA.radius)) * 1e6:.2f} µm")
print(f"Expansion : {float(jnp.max(resultA.radius)) / eomA.R0:.2f}×")

resultB = jax.jit(run_simulation)(
    eomB,
    pulse,
    save_spec=SaveSpec(num_samples=1024),
    window_s=20e-6,
)

print(f"Converged : {bool(resultB.converged)}")
print(f"R0        : {eomB.R0 * 1e6:.2f} µm")
print(f"R_max     : {float(jnp.max(resultB.radius)) * 1e6:.2f} µm")
print(f"R_min     : {float(jnp.min(resultB.radius)) * 1e6:.2f} µm")
print(f"Expansion : {float(jnp.max(resultB.radius)) / eomB.R0:.2f}×")

plt.plot(resultA.ts * 1e6, resultA.radius * 1e6, label="Rayleigh-Plesset")
plt.plot(resultB.ts * 1e6, resultB.radius * 1e6, label="Keller-Miksis")
plt.xlabel("Time (µs)")
plt.ylabel("Radius (µm)")
plt.title("Bubble Radius vs Time")
plt.grid()
plt.show()
