"""07 Cavitation Regimes

Comparing Stable vs Inertial Cavitation.

- Stable Cavitation: Periodic, low-amplitude oscillations. 
  The bubble remains intact for many cycles.
- Inertial Cavitation: Rapid expansion followed by a violent 
  collapse. The collapse is driven by the inertia of the 
  surrounding liquid and can lead to bubble fragmentation 
  and high temperatures/pressures.

We demonstrate this by varying the acoustic pressure amplitude.
"""

import jax
import matplotlib.pyplot as plt
from jbubble import run_simulation
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import VanDerWaalsGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import LipidShell, MarmottantSurfaceTension
from jbubble.pulse import ToneBurst, Sine, HannEnvelope
from jbubble.solver import SaveSpec

# 1. Physics setup (using Keller-Miksis for better collapse accuracy)
eom = KellerMiksis(
    gas=VanDerWaalsGas(gamma=1.4, h_frac=1/5.61),
    shell=LipidShell(
      sigma=MarmottantSurfaceTension(
          R_buckle_ratio=0.99,
          chi=0.38,
          sigma_rupture=72e-3,
      ),
      kappa_s=2.4e-9,
    ),
    medium=NewtonianMedium(mu=0.001),
    R0=2.5e-6,        # 5 micron bubble
    P_amb=101325.0,
    rho_L=998.0,
    c_L=1500.0,     # Speed of sound in water
)

# 2. Define driving pressures
low_pressure = 40e3   # 40 kPa -> Stable
high_pressure = 350e3 # 150 kPa -> Inertial

# Create pulses
pulse_stable = ToneBurst(freq=1e6, pressure=low_pressure, shape=Sine(), cycle_num=5, envelope=HannEnvelope())
pulse_inertial = ToneBurst(freq=1e6, pressure=high_pressure, shape=Sine(), cycle_num=5, envelope=HannEnvelope())

# 3. Use JAX parallelization (vmap) to run both at once
# We can define a helper and vmap it over the pulses.
def sim(p):
    return run_simulation(eom, p, save_spec=SaveSpec(num_samples=2000), t_span=(0, 15e-6))

v_sim = jax.jit(jax.vmap(sim))
results = v_sim(jax.tree.map(lambda x, y: jax.numpy.stack([x, y]), pulse_stable, pulse_inertial))

res_stable = jax.tree.map(lambda x: x[0], results)
res_inertial = jax.tree.map(lambda x: x[1], results)

# 4. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Stable Cavitation
ax1.plot(res_stable.ts * 1e6, res_stable.radius * 1e6, color="blue", lw=2)
ax1.set_ylabel("Radius (µm)")
ax1.set_title(f"Stable Cavitation ({low_pressure/1e3} kPa)")
ax1.grid(True, alpha=0.3)

# Inertial Cavitation
ax2.plot(res_inertial.ts * 1e6, res_inertial.radius * 1e6, color="red", lw=2)
ax2.set_xlabel("Time (µs)")
ax2.set_ylabel("Radius (µm)")
ax2.set_title(f"Inertial Cavitation ({high_pressure/1e3} kPa)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
