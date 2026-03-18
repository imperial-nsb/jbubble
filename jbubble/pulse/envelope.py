"""Envelope (window) functions for acoustic pulses."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp


class Envelope(eqx.Module, abc.ABC):
    """Window function mapping relative time *tau* to a scale in [0, 1].

    Called as ``envelope(tau, duration)`` where *tau* = t − initial_time.
    Returns 0 outside [0, duration].
    """

    @abc.abstractmethod
    def __call__(self, tau: jax.Array, duration: float) -> jax.Array: ...


class RectangularEnvelope(Envelope):
    """Hard on/off gating — 1 inside [0, duration], 0 outside."""

    def __call__(self, tau: jax.Array, duration: float) -> jax.Array:
        return jnp.where((tau >= 0) & (tau <= duration), 1.0, 0.0)


class HannEnvelope(Envelope):
    """Hann (raised-cosine) window for smooth on/off transitions."""

    def __call__(self, tau: jax.Array, duration: float) -> jax.Array:
        in_window = (tau >= 0) & (tau <= duration)
        hann = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau / duration))
        return jnp.where(in_window, hann, 0.0)


class SoftRectangularEnvelope(Envelope):
    """Smooth approximation to a rectangular window using sigmoid transitions.

    Replaces the hard on/off step of :class:`RectangularEnvelope` with
    smooth sigmoid ramps, keeping ``dp_ac/dt`` continuous everywhere.
    This is the preferred envelope when a near-rectangular window is needed
    for gradient-based parameter fitting via the adjoint method.

    The window value is::

        w(τ) = σ(τ / k) · σ((T − τ) / k),   k = T / steepness

    where ``σ`` is the logistic sigmoid and ``T`` is the pulse duration.
    The plateau is flat to within ``2·exp(−steepness/2)`` of 1.0.

    Parameters
    ----------
    steepness : float
        Controls transition sharpness.  Transitions span roughly
        ``4·T / steepness`` in time (±2σ).  Default 100 gives transitions
        of ≈ 4% of the pulse duration — imperceptible for bursts of 5+ cycles.
    """

    steepness: float = 100.0

    def __call__(self, tau: jax.Array, duration: float) -> jax.Array:
        k = duration / self.steepness
        return jax.nn.sigmoid(tau / k) * jax.nn.sigmoid((duration - tau) / k)


class TukeyEnvelope(Envelope):
    """Tukey (tapered cosine) window — flat in the middle, cosine tapers.

    Parameters
    ----------
    alpha : float
        Fraction of the window inside the cosine tapers.
        ``alpha = 0`` → rectangular, ``alpha = 1`` → Hann.
    """

    alpha: float = 0.5

    def __call__(self, tau: jax.Array, duration: float) -> jax.Array:
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
