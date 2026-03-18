"""03 Shell Models

Comparing different shell (coating) models.

We contrast a simple uncoated bubble (NoShell) with a lipid-coated
microbubble (LipidShell) using the Marmottant surface tension law.
The Marmottant law captures buckling and rupture behaviors.
"""

import jax
import matplotlib.pyplot as plt
from jbubble import run_simulation
from jbubble.bubble.eom import ModifiedRayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell, LipidShell, MarmottantSurfaceTension
from jbubble.pulse import ToneBurst, Sine, HannEnvelope
from jbubble.solver import SaveSpec

# Common setup
R0 = 2e-6
pressure = 150e3  # 150 kPa
freq = 1e6
window = 15e-6

# 1. Uncoated Bubble (No Shell)
# Only water-air surface tension (0.072 N/m)
eom_no_shell = ModifiedRayleighPlesset(
    gas=PolytropicGas(gamma=1.07),
    shell=NoShell(sigma=0.072),
    medium=NewtonianMedium(mu=0.001),
    R0=R0,
    P_amb=101325.0,
    rho_L=998.0,
    c_L=1500.0,
)

# 2. Lipid-Coated Bubble (Marmottant Model)
# We define a piecewise surface tension (buckling, elastic, rupture regimes)
# and a shell viscosity (kappa_s).
sigma_marm = MarmottantSurfaceTension(R_buckle_ratio=0.98, chi=0.5, sigma_rupture=0.072)

eom_lipid = ModifiedRayleighPlesset(
    gas=PolytropicGas(gamma=1.07),
    shell=LipidShell(sigma=sigma_marm, kappa_s=1e-9),
    medium=NewtonianMedium(mu=0.001),
    R0=R0,
    P_amb=101325.0,
    rho_L=998.0,
    c_L=1500.0,
)

# Define driving pulse
pulse = ToneBurst(
    freq=freq, pressure=pressure, shape=Sine(), cycle_num=8, envelope=HannEnvelope()
)

# Run simulations
# JIT-compiling the simulation function
sim_fn = jax.jit(run_simulation)

res_no_shell = sim_fn(
    eom_no_shell, pulse, save_spec=SaveSpec(num_samples=1000), t_max=window
)
res_lipid = sim_fn(
    eom_lipid, pulse, save_spec=SaveSpec(num_samples=1000), t_max=window
)

# Visualize comparison
plt.figure(figsize=(10, 5))
plt.plot(
    res_no_shell.ts * 1e6,
    res_no_shell.radius / R0,
    label="Uncoated",
    alpha=0.7,
    ls="--",
)
plt.plot(
    res_lipid.ts * 1e6, res_lipid.radius / R0, label="Lipid Shell (Marmottant)", lw=2
)

plt.xlabel("Time (µs)")
plt.ylabel("Normalized Radius (R/R0)")
plt.title("Effect of Shell Coating on Acoustic Response")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
