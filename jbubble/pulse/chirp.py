"""Chirp pulse — frequency sweep over time."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .base import Envelope, Pulse, RectangularEnvelope


class ChirpPulse(Pulse):
    """Linear or exponential frequency-sweep pulse.

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
    initial_time : float
        Time at which the chirp begins [s].  Default: 0.
    envelope : Envelope
        Window applied to the chirp.  Default: ``RectangularEnvelope()``.
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
    initial_time: float = 0.0
    envelope: Envelope = eqx.field(default_factory=RectangularEnvelope)
    method: str = eqx.field(default="linear", static=True)

    @property
    def duration(self) -> float:
        return self.sweep_duration

    @property
    def t_end(self) -> float:
        return self.initial_time + 2.0 * self.duration

    def __call__(self, t: jax.Array) -> jax.Array:
        tau = t - self.initial_time
        f0, f1, T = self.freq_start, self.freq_end, self.sweep_duration

        if self.method == "linear":
            # Instantaneous freq: f(tau) = f0 + (f1 - f0) * tau / T
            # Phase: integral = 2*pi * (f0*tau + (f1 - f0)*tau^2 / (2T))
            phase = 2.0 * jnp.pi * (f0 * tau + (f1 - f0) * tau**2 / (2.0 * T))
        else:
            # Exponential chirp: f(tau) = f0 * (f1/f0)^(tau/T)
            # Phase: integral = 2*pi * f0 * T * ((f1/f0)^(tau/T) - 1) / ln(f1/f0)
            ratio = f1 / f0
            phase = (
                2.0
                * jnp.pi
                * f0
                * T
                * (jnp.power(ratio, tau / T) - 1.0)
                / jnp.log(ratio)
            )

        val = jnp.sin(phase)
        window = self.envelope(tau, self.sweep_duration)
        return val * self.pressure * window
