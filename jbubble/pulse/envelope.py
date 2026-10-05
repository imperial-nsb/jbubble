"""Envelope (window) functions for acoustic pulses."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "Envelope",
    "NoEnvelope",
    "RectangularEnvelope",
    "HannEnvelope",
    "SoftRectangularEnvelope",
    "TukeyEnvelope",
]


class Envelope(eqx.Module, abc.ABC):
    """Window function that maps relative time `tau` to a scale in [0, 1].

    A pulse calls it as `envelope(tau, duration)`, where
    `tau = t - t_start` and `t_start` is the pulse's
    [`t_start`][jbubble.pulse.base.Pulse.t_start]. It returns 0 outside
    [0, `duration`], except for two envelopes:
    [`SoftRectangularEnvelope`][jbubble.pulse.envelope.SoftRectangularEnvelope],
    whose sigmoid tails decay smoothly outside the window, and
    [`NoEnvelope`][jbubble.pulse.envelope.NoEnvelope], which is 1 at all
    times.
    """

    @abc.abstractmethod
    def __call__(self, tau: jax.Array, duration: ArrayLike) -> jax.Array: ...


class NoEnvelope(Envelope):
    """Identity window: 1 at all times, so the signal passes unchanged.

    It is the default envelope of [`Summed`][jbubble.pulse.base.Summed],
    whose children already apply their own envelopes. To remove the
    window of a windowed sum, call `pulse.windowed(NoEnvelope())`.
    """

    def __call__(self, tau: jax.Array, duration: ArrayLike) -> jax.Array:
        return jnp.ones_like(tau)


class RectangularEnvelope(Envelope):
    """Hard on/off gating: 1 inside [0, `duration`], 0 outside."""

    def __call__(self, tau: jax.Array, duration: ArrayLike) -> jax.Array:
        return jnp.where((tau >= 0) & (tau <= duration), 1.0, 0.0)


class HannEnvelope(Envelope):
    r"""Hann (raised-cosine) window for smooth on/off transitions.

    $$
    w(\tau) = \frac{1}{2}\left[1 - \cos\left(\frac{2\pi\tau}{T}\right)\right],
    \qquad 0 \le \tau \le T,
    $$

    and 0 outside, where $T$ is the pulse duration.
    """

    def __call__(self, tau: jax.Array, duration: ArrayLike) -> jax.Array:
        in_window = (tau >= 0) & (tau <= duration)
        hann = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau / duration))
        return jnp.where(in_window, hann, 0.0)


class SoftRectangularEnvelope(Envelope):
    r"""Smooth approximation to a rectangular window with sigmoid transitions.

    Replaces the hard on/off step of
    [`RectangularEnvelope`][jbubble.pulse.envelope.RectangularEnvelope] with
    smooth sigmoid ramps, which keeps $\mathrm{d}p_\text{ac}/\mathrm{d}t$
    continuous everywhere. This is the preferred envelope when you need a
    near-rectangular window for gradient-based parameter fitting with the
    adjoint method.

    The window value is

    $$
    w(\tau) = S\left(\frac{\tau}{k}\right) S\left(\frac{T - \tau}{k}\right),
    \qquad k = \frac{T}{\text{steepness}},
    $$

    where $S$ is the logistic sigmoid and $T$ is the pulse duration. The
    plateau is flat to within $2\exp(-\text{steepness}/2)$ of 1.

    Parameters
    ----------
    steepness : float
        Controls the transition sharpness. Each transition spans roughly
        $4T/\text{steepness}$ in time, from $-2k$ to $+2k$. The default,
        `100.0`, gives transitions of about 4% of the pulse duration, which
        is imperceptible for bursts of five or more cycles.
    """

    steepness: float = 100.0

    def __call__(self, tau: jax.Array, duration: ArrayLike) -> jax.Array:
        k = duration / self.steepness
        return jax.nn.sigmoid(tau / k) * jax.nn.sigmoid((duration - tau) / k)


class TukeyEnvelope(Envelope):
    """Tukey (tapered cosine) window: flat in the middle, with cosine tapers.

    Parameters
    ----------
    alpha : float
        Fraction of the window inside the cosine tapers. Must be greater
        than 0. As `alpha` approaches 0, the window approaches a
        rectangular window; for a hard gate, use
        [`RectangularEnvelope`][jbubble.pulse.envelope.RectangularEnvelope].
        `alpha = 1` gives a Hann window. Default: `0.5`.
    """

    alpha: float = 0.5

    def __call__(self, tau: jax.Array, duration: ArrayLike) -> jax.Array:
        in_window = (tau >= 0) & (tau <= duration)
        frac = tau / duration  # normalised position in [0, 1]

        # Lower taper: 0 <= frac < alpha/2
        lower = 0.5 * (
            1.0 + jnp.cos(2.0 * jnp.pi / self.alpha * (frac - self.alpha / 2.0))
        )
        # Upper taper: 1 - alpha/2 < frac <= 1
        upper = 0.5 * (
            1.0 + jnp.cos(2.0 * jnp.pi / self.alpha * (frac - 1.0 + self.alpha / 2.0))
        )

        val = jnp.where(frac < self.alpha / 2.0, lower, 1.0)
        val = jnp.where(frac > 1.0 - self.alpha / 2.0, upper, val)
        return jnp.where(in_window, val, 0.0)
