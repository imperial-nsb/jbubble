"""Chirp pulse: a frequency sweep over time."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .base import Pulse
from .shapes import PulseShape, Sine

__all__ = ["ChirpSweep", "LinearSweep", "ExponentialSweep", "ChirpPulse"]


class ChirpSweep(eqx.Module):
    r"""Abstract phase law for a frequency sweep.

    Subclasses compute the *accumulated phase* $\Phi(\tau)$ [rad] as a
    function of the elapsed time $\tau$, the start and end frequencies,
    and the sweep duration. [`ChirpPulse`][jbubble.pulse.chirp.ChirpPulse]
    then evaluates the carrier
    [`PulseShape`][jbubble.pulse.shapes.PulseShape] at that phase.
    """

    @abc.abstractmethod
    def __call__(
        self,
        tau: jax.Array,
        freq_start: ArrayLike,
        freq_end: ArrayLike,
        duration: ArrayLike,
    ) -> jax.Array:
        r"""Return the accumulated phase $\Phi(\tau)$ [rad]."""
        ...


class LinearSweep(ChirpSweep):
    r"""Linear (constant-rate) frequency sweep.

    $$
    \Phi(\tau) = 2\pi\left[f_0 \tau + \frac{(f_1 - f_0)\,\tau^2}{2T}\right]
    $$

    where $f_0$ is `freq_start`, $f_1$ is `freq_end`, and $T$ is the sweep
    duration.
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
    r"""Exponential (geometric) frequency sweep.

    $$
    \Phi(\tau) = \frac{2\pi f_0 T \left(r^{\tau/T} - 1\right)}{\ln r},
    \qquad r = \frac{f_1}{f_0},
    $$

    where $f_0$ is `freq_start`, $f_1$ is `freq_end`, and $T$ is the sweep
    duration.
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
    r"""Frequency-sweep pulse with a composable carrier shape and sweep law.

    The `shape` field sets the carrier waveform (default: sine); the
    `sweep` field sets how the instantaneous frequency varies with time
    (default: linear). Both are `eqx.Module` leaves, so swapping either
    keeps the same computational graph and doesn't force a re-trace.

    The pulse evaluates the shape at the accumulated phase $\Phi(\tau)$,
    with $\tau = t - t_0$, so any
    [`PulseShape`][jbubble.pulse.shapes.PulseShape] works as a carrier.

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
        Carrier waveform shape. Default:
        [`Sine()`][jbubble.pulse.shapes.Sine].
    sweep : ChirpSweep
        Phase law. Default:
        [`LinearSweep()`][jbubble.pulse.chirp.LinearSweep].

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
