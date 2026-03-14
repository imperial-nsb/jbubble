#!/usr/bin/env python3
"""
Test script for multi-model bubble simulations.
"""

import matplotlib.pyplot as plt

from jbubble import shapes
from jbubble.bubble import (
    Marmottant,
    MarmottantGompertz,
    RayleighPlesset,
    KellerMiksisGompertz,
    SphericalConfinement,
    LeightonGompertz,
)
from jbubble.pulse import Pulse
from jbubble.solver import SaveSpec
from jbubble.simulation import run_simulation


bubble = Marmottant(R0=4e-6)  # R0 = 4 µm
bubble_gompertz = MarmottantGompertz(R0=4e-6)  # R0 = 4 µm

pulse = Pulse(
    shape=shapes.Sine(),
    freq=300e3,
    pressure=50e3,
    cycle_num=20,
    initial_time=1e-6,
    apply_hann=False,
)
save_spec = SaveSpec(num_samples=1000)

result_rayleigh_plesset = run_simulation(
    bubble=RayleighPlesset(R0=2.5e-6),  # R0 = 4 µm
    pulse=pulse,
    save_spec=save_spec,
    window_s=20e-6,
    dt0=1e-3,
    max_steps=10_000,
    progress=True,
)
result_marmottant = run_simulation(
    bubble=bubble,
    pulse=pulse,
    save_spec=save_spec,
    window_s=20e-6,
    dt0=1e-3,
    max_steps=10_000,
    progress=True,
)

result_gompertz = run_simulation(
    bubble=bubble_gompertz,
    pulse=pulse,
    save_spec=save_spec,
    window_s=20e-6,
    dt0=1e-3,
    max_steps=10_000,
    progress=True,
)

result_keller_miksis_gompertz = run_simulation(
    bubble=KellerMiksisGompertz(R0=4e-6),  # R0 = 4 µm
    pulse=pulse,
    save_spec=save_spec,
    window_s=20e-6,
    dt0=1e-3,
    max_steps=10_000,
    progress=True,
)

result_leighton_gompertz = run_simulation(
    bubble=LeightonGompertz(R0=4e-6),  # R0 = 4 µm
    pulse=pulse,
    save_spec=save_spec,
    window_s=20e-6,
    dt0=1e-3,
    max_steps=10_000,
    progress=True,
)

result_spherical_confinement = run_simulation(
    bubble=SphericalConfinement(R0=4e-6, vessel_radius=10e-6),  # R0 = 4 µm
    pulse=pulse,
    save_spec=save_spec,
    window_s=20e-6,
    dt0=1e-3,
    max_steps=10_000,
    progress=True,
)


# # Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes

ax6.plot(
    result_spherical_confinement.ts * 1e6,
    result_spherical_confinement.radius * 1e6,
    label="Spherical Confinement",
    lw=2.5,
    color="#8c564b",
)
ax6.set_xlabel("Time [µs]", fontsize=12)
ax6.set_ylabel("Radius [µm]", fontsize=12)
ax6.set_title("Spherical Confinement Bubble Dynamics", fontsize=14)
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=11)

ax5.plot(
    result_leighton_gompertz.ts * 1e6,
    result_leighton_gompertz.radius * 1e6,
    label="Leighton-Gompertz",
    lw=2.5,
    color="#9467bd",
)
ax5.set_xlabel("Time [µs]", fontsize=12)
ax5.set_ylabel("Radius [µm]", fontsize=12)
ax5.set_title("Leighton-Gompertz Bubble Dynamics", fontsize=14)
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=11)

ax4.plot(
    result_keller_miksis_gompertz.ts * 1e6,
    result_keller_miksis_gompertz.radius * 1e6,
    label="Keller-Miksis-Gompertz",
    lw=2.5,
    color="#d62728",
)
ax4.set_xlabel("Time [µs]", fontsize=12)
ax4.set_ylabel("Radius [µm]", fontsize=12)
ax4.set_title("Keller-Miksis-Gompertz Bubble Dynamics", fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=11)


ax2.plot(
    result_marmottant.ts * 1e6,
    result_marmottant.radius * 1e6,
    label="Marmottant",
    lw=2.5,
    color="#1f77b4",
)
ax2.set_xlabel("Time [µs]", fontsize=12)
ax2.set_ylabel("Radius [µm]", fontsize=12)
ax2.set_title("Marmottant Bubble Dynamics", fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

ax3.plot(
    result_gompertz.ts * 1e6,
    result_gompertz.radius * 1e6,
    label="Marmottant-Gompertz",
    lw=2.5,
    color="#ff7f0e",
)
ax3.set_xlabel("Time [µs]", fontsize=12)
ax3.set_ylabel("Radius [µm]", fontsize=12)
ax3.set_title("Marmottant-Gompertz Bubble Dynamics", fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

ax1.plot(
    result_rayleigh_plesset.ts * 1e6,
    result_rayleigh_plesset.radius * 1e6,
    label="Rayleigh-Plesset",
    lw=2.5,
    color="#2ca02c",
)
ax1.set_xlabel("Time [µs]", fontsize=12)
ax1.set_ylabel("Radius [µm]", fontsize=12)
ax1.set_title("Rayleigh-Plesset Bubble Dynamics", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

plt.tight_layout()
plt.show()
