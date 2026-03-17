"""08 Acoustic Emissions

Computing the radiated acoustic pressure from a cavitating bubble.

In most experiments we don't observe the bubble radius directly —
instead we measure the acoustic emission at a hydrophone some
distance away.  jbubble provides emission models that convert a
solved bubble trajectory into a pressure signal at an arbitrary
field point.

This example demonstrates:
  - Computing monopole emission at a single distance
  - Comparing IncompressibleMonopole and QuasiAcoustic models
  - Computing emission at multiple distances via jax.vmap
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jbubble import run_simulation
from jbubble.acoustics import IncompressibleMonopole, QuasiAcoustic
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.solver import SaveSpec

# 1. Simulate a bubble
eom = KellerMiksis(
    gas=PolytropicGas(gamma=1.4),
    shell=NoShell(sigma=0.072),
    medium=NewtonianMedium(mu=0.001),
    R0=2e-6,
    P_amb=101325.0,
    rho_L=998.0,
    c_L=1500.0,
)
pulse = ToneBurst(freq=1e6, pressure=200e3, shape=Sine(), cycle_num=5)

result = jax.jit(run_simulation)(
    eom,
    pulse,
    save_spec=SaveSpec(num_samples=2048),
    t_span=(0, 10e-6),
)

# 2. Compute emission at a single distance
monopole = IncompressibleMonopole(rho_L=998.0)
r_sensor = 0.01  # 1 cm from bubble

p_rad = monopole(result, r=r_sensor)

# 3. Compare monopole and quasi-acoustic models
quasi = QuasiAcoustic(rho_L=998.0, c_L=1500.0)
p_qa = quasi(result, r=r_sensor)

# 4. Compute emission at multiple distances using vmap
distances = jnp.array([0.001, 0.005, 0.01, 0.05])  # 1 mm to 5 cm
p_all = jax.vmap(lambda r: monopole(result, r))(distances)
# p_all has shape (4, 2048) — one time series per distance

# 5. Plot results
t_us = result.ts * 1e6  # time in microseconds

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Panel 1: bubble radius
axes[0].plot(t_us, result.radius * 1e6, lw=2, color="navy")
axes[0].set_ylabel("Radius (µm)")
axes[0].set_title("Bubble Dynamics and Acoustic Emissions")
axes[0].grid(True, alpha=0.3)

# Panel 2: monopole vs quasi-acoustic at 1 cm
axes[1].plot(t_us, p_rad, lw=1.5, label="Incompressible monopole")
axes[1].plot(t_us, p_qa, lw=1.5, ls="--", label="Quasi-acoustic")
axes[1].set_ylabel("Pressure (Pa)")
axes[1].set_title(f"Emission at r = {r_sensor * 100:.0f} cm")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Panel 3: monopole at multiple distances
for i, d in enumerate(distances):
    label = f"r = {float(d) * 1000:.0f} mm"
    axes[2].plot(t_us, p_all[i], lw=1.5, label=label)
axes[2].set_xlabel("Time (µs)")
axes[2].set_ylabel("Pressure (Pa)")
axes[2].set_title("Emission vs Distance (1/r scaling)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
