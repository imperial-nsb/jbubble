"""Pulse combinators — scale, sum, and window arbitrary pulses."""

import jax
import jax.numpy as jnp

from .base import Envelope, Pulse


class Scaled(Pulse):
    """Amplitude-scaled version of another pulse.

    Parameters
    ----------
    pulse : Pulse
        The pulse to scale.
    factor : float
        Multiplicative factor.
    """

    pulse: Pulse
    factor: float

    @property
    def duration(self) -> float:
        return self.pulse.duration

    @property
    def t_end(self) -> float:
        return self.pulse.t_end

    def __call__(self, t: jax.Array) -> jax.Array:
        return self.factor * self.pulse(t)


class Summed(Pulse):
    """Additive superposition of multiple pulses.

    Parameters
    ----------
    pulses : tuple[Pulse, ...]
        Pulses to sum.  Must be a tuple (not a list) for Equinox
        PyTree compatibility.
    """

    pulses: tuple[Pulse, ...]

    @property
    def duration(self) -> float:
        return max(p.duration for p in self.pulses)

    @property
    def t_end(self) -> float:
        return max(p.t_end for p in self.pulses)

    def __call__(self, t: jax.Array) -> jax.Array:
        return jnp.sum(jnp.array([p(t) for p in self.pulses]))


class Windowed(Pulse):
    """Apply an envelope to any pulse.

    Parameters
    ----------
    pulse : Pulse
        The pulse to window.
    envelope : Envelope
        The envelope to apply.
    """

    pulse: Pulse
    envelope: Envelope

    @property
    def duration(self) -> float:
        return self.pulse.duration

    @property
    def t_end(self) -> float:
        return self.pulse.t_end

    def __call__(self, t: jax.Array) -> jax.Array:
        return self.pulse(t) * self.envelope(t, self.pulse.duration)
