"""09 Custom Pulse Shapes

Demonstrates how to extend jbubble with your own pulse shapes by subclassing
FourierPulseShape or PulseShape directly.

Two approaches are shown:

  1. **Custom Fourier shape** — define term() and norm_factor; the base class
     handles the vmap summation and the (optional) dc_offset.

  2. **Rectangular** — the built-in parameterised rectangular wave. All
     named variants from the literature (monopolar, asymmetric duty cycles,
     NegPos phasing) reduce to a single class instantiation.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jbubble.pulse.shapes import FourierPulseShape, Rectangular

# ---------------------------------------------------------------------------
# 1.  Custom Fourier shapes — just define term() and norm_factor
# ---------------------------------------------------------------------------

class SlantedSine(FourierPulseShape):
    """Clausen-function waveform: Σ (-1)^m / m^2 · sin(2π m f t)."""

    def term(self, m, t, freq, phase):
        return ((-1) ** m / (m**2)) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)

    @property
    def norm_factor(self):
        return 1.0149  # approximate max of Clausen function Cl₂

    @property
    def name(self):
        return "slanted_sine"


class Asymmetrical(FourierPulseShape):
    """Mixed cosine + sine Fourier series giving a pronounced asymmetry."""

    def term(self, m, t, freq, phase):
        return (1.0 / (m**2)) * jnp.cos(2.0 * jnp.pi * m * freq * t - m * phase) - (
            1.0 / m
        ) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)

    @property
    def norm_factor(self):
        return -((jnp.pi**2) / 6.0 + jnp.pi / 2.0)

    @property
    def name(self):
        return "asymmetrical"


# ---------------------------------------------------------------------------
# 2.  Named rectangular variants via Rectangular
# ---------------------------------------------------------------------------

# Classic duty-cycle waves (high=+1 for `duty` fraction of cycle)
rect_25 = Rectangular(duty=0.25)          # +1 for 25%, -1 for 75%
rect_75 = Rectangular(duty=0.75)          # +1 for 75%, -1 for 25%

# NegPos: -1 first half, +1 second half (phase_offset = π shifts window to back)
square_negpos = Rectangular(duty=0.5, phase_offset=jnp.pi)
rect25_negpos = Rectangular(duty=0.25, phase_offset=0.5 * jnp.pi)
rect75_negpos = Rectangular(duty=0.75, phase_offset=1.5 * jnp.pi)

# Monopolar (0 / −1) waveforms — rest period at end of cycle
mono95 = Rectangular(duty=0.05, high_level=0.0, low_level=-1.0, phase_offset=1.9 * jnp.pi)
mono99 = Rectangular(duty=0.01, high_level=0.0, low_level=-1.0, phase_offset=1.98 * jnp.pi)

# Thin positive pulse at end of cycle (±1 levels, 5% high)
rect95 = Rectangular(duty=0.05, phase_offset=1.9 * jnp.pi)

# ---------------------------------------------------------------------------
# Plot everything
# ---------------------------------------------------------------------------

fourier_shapes = [SlantedSine(), Asymmetrical()]
rect_shapes = [
    (rect_25,      "Rectangular(duty=0.25)"),
    (rect_75,      "Rectangular(duty=0.75)"),
    (square_negpos,"Rectangular(duty=0.5, phase_offset=π)   [NegPos]"),
    (rect25_negpos,"Rectangular(duty=0.25, phase_offset=π/2) [NegPos]"),
    (rect75_negpos,"Rectangular(duty=0.75, phase_offset=3π/2)[NegPos]"),
    (mono95,       "Rectangular(duty=0.05, 0/−1, phase=1.9π) [Mono95]"),
    (mono99,       "Rectangular(duty=0.01, 0/−1, phase=1.98π)[Mono99]"),
    (rect95,       "Rectangular(duty=0.05, phase=1.9π)        [Rect95]"),
]

freq = 1.0
t = jnp.linspace(0, 2.0 / freq, 1000)

n_total = len(fourier_shapes) + len(rect_shapes)
n_cols = 2
n_rows = (n_total + 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.2))
axes = axes.flatten()

idx = 0
for shape in fourier_shapes:
    ax = axes[idx]
    ax.plot(t, shape(t, freq, 0.0, 0.0), linewidth=1.5)
    ax.set_title(shape.name, fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("t (s)", fontsize=8)
    idx += 1

for shape, label in rect_shapes:
    ax = axes[idx]
    ax.plot(t, shape(t, freq, 0.0, 0.0), linewidth=1.5, color="tab:orange")
    ax.set_title(label, fontsize=8, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("t (s)", fontsize=8)
    idx += 1

for ax in axes[idx:]:
    ax.set_visible(False)

fig.suptitle("Custom pulse shapes  (examples/09_custom_pulse_shapes.py)", fontsize=11)
plt.tight_layout()
plt.savefig("custom_pulse_shapes.png", dpi=150, bbox_inches="tight")
print(f"Saved custom_pulse_shapes.png  ({n_total} shapes)")
plt.show()
