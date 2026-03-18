"""00 Presets — Quick-Start Simulations

The fastest way to run a microbubble simulation: pick a preset,
optionally override a parameter, and go.

Three presets are available:

  free_bubble        — uncoated gas bubble (no shell)
  lipid_bubble       — thin lipid shell (Marmottant 2005)
  thick_shell_bubble — polymer thick shell (Church 1995)

Each returns a (eom, pulse) pair ready for run_simulation.
See 01_basic_simulation.py to understand how to assemble
these components manually.
"""

import jax
import matplotlib.pyplot as plt

from jbubble import SaveSpec, run_simulation
from jbubble.utils.presets import free_bubble, lipid_bubble, thick_shell_bubble

# --- 1. Build presets ---------------------------------------------------------
# Default parameters represent a 2 µm bubble in water driven at 1 MHz / 100 kPa.
# Override only what you need; everything else stays physically reasonable.

presets = {
    "Free bubble\n(no shell)": free_bubble(),
    "Lipid shell\n(Marmottant)": lipid_bubble(),
    "Thick shell\n(Church)": thick_shell_bubble(),
}

# Overriding a parameter is just a keyword argument:
#   lipid_bubble(R0=3e-6, pressure=200e3)

# --- 2. Run simulations -------------------------------------------------------
results = {
    label: jax.jit(run_simulation)(eom, pulse, save_spec=SaveSpec(num_samples=1000))
    for label, (eom, pulse) in presets.items()
}

# --- 3. Plot ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, (label, result) in zip(axes, results.items(), strict=True):
    ax.plot(result.ts * 1e6, result.radius * 1e6, lw=1.8, color="navy")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Time (µs)")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Radius (µm)")
fig.suptitle("2 µm bubble in water — 1 MHz, 100 kPa", fontsize=12)
plt.tight_layout()
plt.show()
