"""Core abstractions for acoustic driving pulses."""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp


class Pulse(eqx.Module, abc.ABC):
    """Abstract acoustic driving pulse.

    Every ``Pulse`` is callable: ``pulse(t)`` returns the instantaneous
    pressure [Pa] at time ``t``.  Implementations must be JAX-differentiable
    so that equations of motion (e.g. Keller–Miksis) can compute
    ``jax.grad(pulse)(t)``.

    Subclasses must implement :meth:`__call__` and the :attr:`duration`
    property.
    """

    @abc.abstractmethod
    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate instantaneous pressure at time *t* [Pa]."""
        ...

    @property
    @abc.abstractmethod
    def duration(self) -> float:
        """Active pulse duration [s] (excluding any leading silence)."""
        ...

    @property
    def t_end(self) -> float:
        """Suggested simulation end time [s].

        Default: ``2 × duration``.  Override in subclasses that have a
        non-zero ``initial_time`` or other structure.
        """
        return self.duration * 2.0


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


class Envelope(eqx.Module, abc.ABC):
    """Window function mapping relative time *tau* to a scale in [0, 1].

    Called as ``envelope(tau, duration)`` where *tau* = t − initial_time.
    Returns 0 outside [0, duration].
    """

    @abc.abstractmethod
    def __call__(self, tau: jax.Array, duration: float) -> jax.Array:
        ...


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
        lower = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi / self.alpha * (frac - self.alpha / 2.0)))
        # Upper taper: 1 - alpha/2 < frac <= 1
        upper = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi / self.alpha * (frac - 1.0 + self.alpha / 2.0)))

        val = jnp.where(frac < self.alpha / 2.0, lower, 1.0)
        val = jnp.where(frac > 1.0 - self.alpha / 2.0, upper, val)
        return jnp.where(in_window, val, 0.0)
