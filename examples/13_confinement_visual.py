import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Patch

from jbubble import run_simulation
from jbubble.bubble.eom import SphericalConfinement
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import LipidShell, MarmottantSurfaceTension
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.solver import SaveSpec

gas = PolytropicGas(gamma=1.095)
shell = LipidShell(
    sigma=MarmottantSurfaceTension(
        R_buckle_ratio=0.99,
        chi=0.38,
        sigma_rupture=72e-3,
    ),
    kappa_s=2.4e-9,
)
medium = NewtonianMedium(mu=0.001)

eom = SphericalConfinement(
    gas=gas,
    shell=shell,
    medium=medium,
    R0=4e-6,
    P_amb=101325.0,
    rho_L=998.0,
    c_L=1500.0,
    vessel_radius=8e-6,
    vessel_rho=900,
    vessel_E=1e6,
    vessel_nu=0.5,
    vessel_d=1e-6,
    tissue_rho=900,
    tissue_d=1e-6,
)

pulse = ToneBurst(
    freq=100e3,
    pressure=100e3,
    shape=Sine(),
    cycle_num=10,
)

result = jax.jit(run_simulation)(
    eom,
    pulse,
    save_spec=SaveSpec(num_samples=1000),
)

bubble_r = np.array(result.radius * 1e6)
a_inner = np.array(result.vessel_radius * 1e6)
a_outer = a_inner + float(eom.vessel_d * 1e6)
tissue_outer = a_outer + float(eom.tissue_d * 1e6)
time = np.array(result.ts * 1e6)

print(a_inner)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")

Rmax = tissue_outer.max()
ax.set_xlim(-1.2 * Rmax, 1.2 * Rmax)
ax.set_ylim(-1.2 * Rmax, 1.2 * Rmax)
ax.set_title("Bubble Dynamics in a Confined Vessel")
ax.set_xlabel("µm")
ax.set_ylabel("µm")

bubble_patch = plt.Circle(
    (0, 0),
    bubble_r[0],
    facecolor="orange",
    edgecolor="none",
)

lumen_patch = Wedge(
    (0, 0),
    a_inner[0],
    0,
    360,
    width=a_inner[0] - bubble_r[0],
    facecolor="#ffcccc",
    edgecolor="none",
)

vessel_patch = Wedge(
    (0, 0),
    a_outer[0],
    0,
    360,
    width=a_outer[0] - a_inner[0],
    facecolor="#8B0000",
    edgecolor="none",
)

tissue_patch = Wedge(
    (0, 0),
    tissue_outer[0],
    0,
    360,
    width=tissue_outer[0] - a_outer[0],
    facecolor="lightgrey",
    edgecolor="none",
)

ax.add_patch(tissue_patch)
ax.add_patch(vessel_patch)
ax.add_patch(lumen_patch)
ax.add_patch(bubble_patch)

time_text = ax.text(
    0.02,
    0.95,
    "",
    transform=ax.transAxes,
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.6),
)

legend_handles = [
    Patch(facecolor="orange", edgecolor="none", label="Bubble"),
    Patch(facecolor="#ffcccc", edgecolor="none", label="Lumen"),
    Patch(facecolor="#8B0000", edgecolor="none", label="Vessel wall"),
    Patch(facecolor="lightgrey", edgecolor="none", label="Tissue"),
]

ax.legend(
    handles=legend_handles,
    loc="lower right",
    fontsize=8,
    framealpha=0.8,
    borderpad=0.2,
    labelspacing=0.2,
    handlelength=0.7,
    handleheight=0.7,
    markerscale=0.6,
)


def update(i):
    br = float(bubble_r[i])
    ai = float(a_inner[i])
    ao = float(a_outer[i])
    to = float(tissue_outer[i])

    bubble_patch.set_radius(br)
    lumen_patch.set_radius(ai)
    lumen_patch.set_width(ai - br)
    vessel_patch.set_radius(ao)
    vessel_patch.set_width(ao - ai)
    tissue_patch.set_radius(to)
    tissue_patch.set_width(to - ao)

    time_text.set_text(f"t = {time[i]:.2f} µs")

    return bubble_patch, lumen_patch, vessel_patch, tissue_patch, time_text


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(time),
    interval=20,
    blit=True,
)

plt.show()
