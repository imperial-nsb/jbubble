import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from jbubble import run_simulation
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import LipidShell, MarmottantSurfaceTension
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.utils import GridSweep
from jbubble.solver import SaveSpec


# Create a simulation kernel for 3 parameters of interest.
def get_expansion(R0, freq, chi):
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.095),
        shell=LipidShell(
            sigma=MarmottantSurfaceTension(
                R_buckle_ratio=0.99,
                chi=chi,
                sigma_rupture=72e-3,
            ),
            kappa_s=2.4e-9,
        ),
        medium=NewtonianMedium(mu=0.001),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )

    pulse = ToneBurst(freq=freq, pressure=100e3, shape=Sine(), cycle_num=5)

    result = run_simulation(
        eom,
        pulse,
        save_spec=SaveSpec(num_samples=200),
    )

    return jnp.max(result.radius) / R0


# Create utitlities for 3D stacked plot.
def edges_from_centers(centers):
    centers = np.asarray(centers)
    mid = 0.5 * (centers[:-1] + centers[1:])
    left = centers[0] - (mid[0] - centers[0])
    right = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate(([left], mid, [right]))


def run_3d_sweep_and_plot():
    # Parameter ranges
    chi_values = jnp.linspace(0.05, 0.38, 5)  # 6 stacked planes
    radii = jnp.linspace(1e-6, 5e-6, 60)
    freqs = jnp.linspace(150e3, 300e3, 60)

    heats_list = []

    # Sweep each chi slice
    for chi in chi_values:
        gs = GridSweep(
            fn=lambda R0, freq: get_expansion(R0, freq, chi),
            search_space={"R0": radii, "freq": freqs},
            batch_size=256,
        )

        print(f"Running sweep for chi={float(chi):.3f} ...")
        heats_2d = gs.run()  # shape: (len(R0), len(freq))
        heats_list.append(heats_2d)

    heats = np.stack(heats_list, axis=0)  # (n_chi, n_R0, n_freq)

    x = radii * 1e6  # μm
    y = freqs / 1e3  # kHz

    Xe, Ye = np.meshgrid(edges_from_centers(x), edges_from_centers(y), indexing="xy")

    # color normalization
    vmin = float(np.nanmin(heats))
    vmax = float(np.nanmax(heats))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("turbo", 256)

    # Create 3D stacked plane
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for k, chi in enumerate(chi_values):
        Z = np.full_like(Xe, float(chi))

        # convert to edge-aligned grid
        plane_colors = cmap(norm(heats[k].T))  # transpose matches axes

        ax.plot_surface(
            Xe,
            Ye,
            Z,
            facecolors=plane_colors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.8,
        )

    ax.set_xlabel("R0 (µm)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_zlabel("Chi (N/m)")
    ax.set_title("Stacked Planes of Bubble Expansion\n$(R_{max}/R_0)$")

    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)

    # colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.8, label=r"$R_{\max}/R_0$")

    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    run_3d_sweep_and_plot()
