"""Chirp pulse — frequency sweep over time."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .base import Pulse
from .shapes import PulseShape, Sine


class ChirpSweep(eqx.Module):
    """Abstract phase law for a frequency sweep.

    Subclasses compute the *accumulated phase* Φ(τ) [rad] as a function of
    elapsed time τ, start/end frequencies, and sweep duration.  The carrier
    waveform is then evaluated at that phase via the :class:`PulseShape`.
    """

    @abc.abstractmethod
    def __call__(
        self,
        tau: jax.Array,
        freq_start: ArrayLike,
        freq_end: ArrayLike,
        duration: ArrayLike,
    ) -> jax.Array:
        """Return accumulated phase Φ(τ) [rad]."""
        ...


class LinearSweep(ChirpSweep):
    """Linear (constant rate) frequency sweep.

    ::

        Φ(τ) = 2π [f₀ τ + (f₁ − f₀) τ² / (2 T)]
    """

    def __call__(
        self,
        tau: jax.Array,
        freq_start: ArrayLike,
        freq_end: ArrayLike,
        duration: ArrayLike,
    ) -> jax.Array:
        f0, f1, T = (
            jnp.asarray(freq_start),
            jnp.asarray(freq_end),
            jnp.asarray(duration),
        )
        return 2.0 * jnp.pi * (f0 * tau + (f1 - f0) * tau**2 / (2.0 * T))


class ExponentialSweep(ChirpSweep):
    """Exponential (geometric) frequency sweep.

    ::

        Φ(τ) = 2π f₀ T (r^(τ/T) − 1) / ln(r),   r = f₁/f₀
    """

    def __call__(
        self,
        tau: jax.Array,
        freq_start: ArrayLike,
        freq_end: ArrayLike,
        duration: ArrayLike,
    ) -> jax.Array:
        f0, f1, T = (
            jnp.asarray(freq_start),
            jnp.asarray(freq_end),
            jnp.asarray(duration),
        )
        ratio = f1 / f0
        return (
            2.0 * jnp.pi * f0 * T * (jnp.power(ratio, tau / T) - 1.0) / jnp.log(ratio)
        )


class ChirpPulse(Pulse):
    """Frequency-sweep pulse with a composable carrier shape and sweep law.

    The :attr:`shape` determines the carrier waveform (default: sine); the
    :attr:`sweep` determines how instantaneous frequency varies with time
    (default: linear).  Both are :class:`~equinox.Module` leaves — swapping
    either keeps the same computational graph and does not force a re-trace.

    The shape is evaluated at the instantaneous accumulated phase Φ(τ),
    so any :class:`~jbubble.pulse.shapes.PulseShape` works as a carrier.

    Parameters
    ----------
    freq_start : float
        Instantaneous frequency at the start [Hz].
    freq_end : float
        Instantaneous frequency at the end [Hz].
    pressure : float
        Peak pressure amplitude [Pa].
    sweep_duration : float
        Duration of the frequency sweep [s].
    shape : PulseShape
        Carrier waveform shape (default: :class:`~jbubble.pulse.shapes.Sine`).
    sweep : ChirpSweep
        Phase law (default: :class:`LinearSweep`).

    Examples
    --------
    >>> from jbubble.pulse import ChirpPulse, HannEnvelope
    >>> from jbubble.pulse.chirp import ExponentialSweep
    >>> from jbubble.pulse.shapes import Square
    >>> chirp = ChirpPulse(
    ...     freq_start=0.5e6, freq_end=2e6,
    ...     pressure=200e3, sweep_duration=10e-6,
    ...     shape=Square(), sweep=ExponentialSweep(),
    ...     envelope=HannEnvelope(),
    ... )
    """

    freq_start: ArrayLike
    freq_end: ArrayLike
    pressure: ArrayLike
    sweep_duration: ArrayLike
    shape: PulseShape = eqx.field(default_factory=Sine)
    sweep: ChirpSweep = eqx.field(default_factory=LinearSweep)

    @property
    def duration(self) -> float | jax.Array:
        return jnp.asarray(self.sweep_duration)

    def _evaluate(self, t: jax.Array) -> jax.Array:
        tau = t - self.initial_time
        # Accumulated phase at this instant
        phi = self.sweep(tau, self.freq_start, self.freq_end, self.sweep_duration)
        # Evaluate the carrier at phase φ:
        #   shape(t, freq=0, phase=-φ, initial_time=t)
        # → t - initial_time = 0, freq term vanishes, phase offset = φ
        return self.shape(t, 0.0, -phi, t) * self.pressure
