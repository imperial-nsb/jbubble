"""10 Envelope Shapes

Differentiable Microbubble Dynamics.

Envelopes window the tone burst on and off.  The choice matters for
gradient-based fitting: ``dp_ac/dt`` must be continuous everywhere so the
adjoint method can propagate gradients through the ODE solve cleanly.

This example compares all built-in envelopes and plots their time derivatives
to show where discontinuities would corrupt the adjoint gradient.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jbubble.pulse.envelope import (
    HannEnvelope,
    RectangularEnvelope,
    SoftRectangularEnvelope,
    TukeyEnvelope,
)

duration = 5e-6  # 5-cycle burst at 1 MHz
tau = jnp.linspace(-0.5e-6, 5.5e-6, 2000)

envelopes = [
    ("Rectangular", RectangularEnvelope()),
    ("Hann", HannEnvelope()),
    ("Tukey α=0.2", TukeyEnvelope(alpha=0.2)),
    ("Tukey α=0.05", TukeyEnvelope(alpha=0.05)),
    ("SoftRect s=50", SoftRectangularEnvelope(steepness=50)),
    ("SoftRect s=100 (default)", SoftRectangularEnvelope(steepness=100)),
    ("SoftRect s=200", SoftRectangularEnvelope(steepness=200)),
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for label, env in envelopes:
    w = jnp.array([env(t, duration) for t in tau])
    ax1.plot(tau * 1e6, w, label=label, lw=1.5)

for ax in (ax1, ax2):
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.axvline(duration * 1e6, color="k", lw=0.8, ls="--", alpha=0.4)

ax1.set_ylabel("Amplitude")
ax1.set_title("Envelope shapes")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Numerical derivative — shows where the ODE sees impulses in dp_ac/dt
dt = float(tau[1] - tau[0])
for label, env in envelopes:
    w = jnp.array([env(t, duration) for t in tau])
    dw = jnp.gradient(w, dt)
    ax2.plot(tau * 1e6, dw * 1e-6, label=label, lw=1.5)

ax2.set_xlabel("Time (µs)")
ax2.set_ylabel("dw/dt  (µs⁻¹)")
ax2.set_title("Time derivative — spikes here break adjoint gradients")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
