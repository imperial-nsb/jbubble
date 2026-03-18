"""Sampled pulse — arbitrary waveform from discrete data points."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import Pulse


class SampledPulse(Pulse):
    """Acoustic pulse defined by an array of pressure samples.

    Evaluates the pressure at arbitrary times via piecewise-linear
    interpolation (``jnp.interp``).  The inherited ``envelope`` (default
    ``RectangularEnvelope``) gates the signal to
    ``[initial_time, initial_time + duration]``.

    Parameters
    ----------
    ts : jax.Array, shape (N,)
        Sample time points [s].  Must be monotonically increasing.
    pressures : jax.Array, shape (N,)
        Pressure values [Pa] at each sample time.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jbubble.pulse import SampledPulse
    >>> ts = jnp.linspace(0, 10e-6, 1000)
    >>> pressures = 200e3 * jnp.sin(2 * jnp.pi * 1e6 * ts)
    >>> pulse = SampledPulse(ts=ts, pressures=pressures)
    >>> pulse.duration  # 10 µs
    1e-05
    """

    ts: jax.Array
    pressures: jax.Array

    @property
    def duration(self) -> float:
        return float(self.ts[-1] - self.ts[0])

    def _evaluate(self, t: jax.Array) -> jax.Array:
        return jnp.interp(t, self.ts, self.pressures)

    @staticmethod
    def from_uniform(
        pressures: jax.Array, dt: float, initial_time: float = 0.0
    ) -> SampledPulse:
        """Create a ``SampledPulse`` from uniformly-spaced samples.

        Parameters
        ----------
        pressures : jax.Array, shape (N,)
            Pressure values [Pa].
        dt : float
            Time step between samples [s].
        initial_time : float
            Time of the first sample [s].  Default: 0.
        """
        ts = initial_time + jnp.arange(pressures.shape[0]) * dt
        return SampledPulse(ts=ts, pressures=pressures, initial_time=initial_time)
