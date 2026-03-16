"""Chirp pulse — frequency sweep over time."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .base import Pulse


class ChirpPulse(Pulse):
    """Linear or exponential frequency-sweep pulse.

    The ``initial_time`` and ``envelope`` fields are inherited from
    :class:`Pulse` and can be set as keyword arguments.

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
    method : str
        ``"linear"`` (default) or ``"exponential"``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jbubble.pulse import ChirpPulse, HannEnvelope
    >>> chirp = ChirpPulse(freq_start=0.5e6, freq_end=2e6,
    ...                    pressure=200e3, sweep_duration=10e-6,
    ...                    envelope=HannEnvelope())
    """

    freq_start: float
    freq_end: float
    pressure: float
    sweep_duration: float
    method: str = eqx.field(default="linear", static=True)

    @property
    def duration(self) -> float:
        return self.sweep_duration

    def _evaluate(self, t: jax.Array) -> jax.Array:
        tau = t - self.initial_time
        f0, f1, T = self.freq_start, self.freq_end, self.sweep_duration

        if self.method == "linear":
            phase = 2.0 * jnp.pi * (f0 * tau + (f1 - f0) * tau**2 / (2.0 * T))
        else:
            ratio = f1 / f0
            phase = (
                2.0
                * jnp.pi
                * f0
                * T
                * (jnp.power(ratio, tau / T) - 1.0)
                / jnp.log(ratio)
            )

        return jnp.sin(phase) * self.pressure
