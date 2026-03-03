"""Checkpoint workflow: sweep → save → load → heatmap.

Runs a Frequency × R₀ grid sweep with :class:`~jbubble.utils.GridSweep`,
checkpoints the full batched :class:`~jbubble.simulation.SimulationResult`
to disk via :func:`~jbubble.utils.save`, then reloads it via
:func:`~jbubble.utils.load` and visualises the max-expansion-ratio heatmap.

The axis-ordering note: :class:`GridSweep` sorts its parameter keys
alphabetically.  Here ``"R0" < "freq"`` so the grid shape is
``(N_R0, N_freq)`` and ``grid[i, j]`` corresponds to ``(R0[i], freq[j])``.
We store the grid shape and axis values in the checkpoint metadata so the
loaded flat arrays can be reshaped without access to the original sweep.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from jbubble import Pulse, SaveSpec, Units, run_simulation
from jbubble.bubble import MarmottantGompertz
from jbubble.shapes import Sine
from jbubble.utils import GridSweep, load, save

CHECKPOINT = Path("data/sweep_checkpoint.jb")

# ── simulation setup ──────────────────────────────────────────────────────────

units = Units()
save_spec = SaveSpec(num_samples=512)

freq_values = jnp.linspace(0.2e6, 1.5e6, 40)  # Hz
r0_values = jnp.linspace(1.5e-6, 6.0e-6, 40)  # m


def kernel(R0, freq):
    bubble = MarmottantGompertz(R0=R0)
    pulse = Pulse(
        freq=freq,
        pressure=100e3,
        shape=Sine(),
        cycle_num=5,
        initial_time=1e-6,
    )
    return run_simulation(
        bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6
    )


# ── sweep ─────────────────────────────────────────────────────────────────────

gs = GridSweep(kernel, {"R0": r0_values, "freq": freq_values})
print(f"Running {gs.total_points}-point sweep ({gs.grid_shape} grid) ...")

flat = gs.collect()  # SimulationResult with leading dim (N_R0 * N_freq,)
print(f"Sweep done  —  convergence: {float(flat.converged.mean()):.1%}")

# ── save ──────────────────────────────────────────────────────────────────────

# Store the grid structure in metadata so we can reshape after loading without
# needing the GridSweep object.  GridSweep sorts keys alphabetically, so the
# axis order here is R0 (axis 0) then freq (axis 1).
save(
    CHECKPOINT,
    flat,
    metadata={
        "grid_shape": list(gs.grid_shape),  # [N_R0, N_freq]
        "r0_values_m": r0_values.tolist(),
        "freq_values_hz": freq_values.tolist(),
    },
)
print(f"Checkpoint saved → '{CHECKPOINT}/'")

# ── load ──────────────────────────────────────────────────────────────────────

loaded, meta = load(CHECKPOINT)
print(f"Loaded {loaded.radius.shape[0]} simulations from '{CHECKPOINT}/'")

# ── reshape + metric ──────────────────────────────────────────────────────────

grid_shape = meta["grid_shape"]  # (N_R0, N_freq)
r0_um = np.asarray(meta["r0_values_m"]) * 1e6  # µm  (for axis ticks)
freq_mhz = np.asarray(meta["freq_values_hz"]) * 1e-6  # MHz (for axis ticks)

# Max expansion ratio per simulation, then fold back to (N_R0, N_freq).
# grid[i, j] = R_max/R0 at r0_values[i], freq_values[j].
expansion = loaded.radius.max(axis=-1) / loaded.bubble.R0
grid = np.asarray(expansion).reshape(grid_shape)

# ── plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))

# pcolormesh(x, y, C): x varies along columns, y along rows.
# C has shape (N_y, N_x) = (N_R0, N_freq), x = freq, y = R0  ✓
mesh = ax.pcolormesh(
    freq_mhz,
    r0_um,
    grid,
    shading="auto",
    cmap="inferno",
    rasterized=True,
)
fig.colorbar(mesh, ax=ax, label=r"$R_\mathrm{max}\,/\,R_0$")
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel(r"$R_0$ (µm)")
ax.set_title(r"Max expansion ratio — $f$ vs $R_0$  (MarmottantGompertz, 100 kPa)")
plt.tight_layout()
plt.show()
