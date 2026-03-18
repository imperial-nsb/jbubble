"""04 Batch Sweeps

Harnessing JAX for massive parallel simulations.

Because jbubble is built on JAX, we can use 'jax.vmap' to run
thousands of simulations in parallel on a single GPU or CPU.
The 'GridSweep' utility makes it easy to sweep over a
Cartesian product of parameters.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jbubble import run_simulation
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import LipidShell, MarmottantSurfaceTension
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.utils import GridSweep
from jbubble.solver import SaveSpec


# 1. Define the simulation function for the sweep
# This function takes scalar parameters and returns a figure of merit
# (e.g., maximum expansion ratio).
def get_expansion(R0, freq):
    # Setup simple physics for speed
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=LipidShell(
            sigma=MarmottantSurfaceTension(
                R_buckle_ratio=0.99,
                chi=0.38,
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

    # 1 MHz burst
    pulse = ToneBurst(freq=freq, pressure=100e3, shape=Sine(), cycle_num=5)

    # Run the simulation
    # Note: run_simulation is JAX-traceable, so this entire function
    # can be vmapped and JIT-compiled.
    result = run_simulation(
        eom,
        pulse,
        save_spec=SaveSpec(num_samples=200),
        t_max=10e-6,
    )

    # Return the metric we want to plot
    return jnp.max(result.radius) / R0


# 2. Setup the parameter axes
# We sweep over equilibrium radius (R0) and driving pressure.
radii = jnp.linspace(1e-6, 5e-6, 100)
freqs = jnp.linspace(100e3, 800e3, 100)

# 3. Use GridSweep to execute the batch
# This automatically handles vmapping and JIT-compilation of the grid.
gs = GridSweep(
    fn=get_expansion, search_space={"R0": radii, "freq": freqs}, batch_size=256
)

print(f"Running sweep over {gs.total_points} grid points...")
flat_results = gs.collect()
grid_results = gs.reshape(flat_results)

# 4. Visualize the results as a heatmap
plt.figure(figsize=(8, 6))
# grid_results is a 2D array of the metrics returned by get_expansion.
im = plt.pcolormesh(
    radii * 1e6, freqs / 1e3, grid_results.T, shading="auto", cmap="viridis"
)
plt.colorbar(im, label="Max Expansion Factor (R_max / R0)")
plt.xlabel("Equilibrium Radius R0 (µm)")
plt.ylabel("Driving Frequency (kHz)")
plt.title("Parameter Sweep: Bubble Expansion Map")
plt.grid(True, alpha=0.3, ls=":")
plt.tight_layout()
plt.show()
