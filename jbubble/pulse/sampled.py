"""Sampled pulse: an arbitrary waveform from discrete data points."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .base import Pulse, _concrete

__all__ = ["SampledPulse"]


class SampledPulse(Pulse):
    r"""Acoustic pulse defined by an array of pressure samples.

    Evaluates the pressure at arbitrary times with piecewise-linear
    interpolation (`jnp.interp`), which holds the first and last sample
    values outside `ts`:

    $$
    p(t) = \operatorname{interp}(t;\ t_s, p_s)\,
    w\bigl(t - t_s[0],\ t_s[N-1] - t_s[0]\bigr)
    $$

    where $t_s$ is `ts`, $p_s$ is `pressures`, and $w$ is `envelope`.

    The sample times are absolute, so the active window runs from the
    first sample to the last: [`t_start`][jbubble.pulse.base.Pulse.t_start]
    is `ts[0]`, and `duration` is `ts[-1] - ts[0]`. To delay the waveform,
    shift `ts`. The `initial_time` field doesn't move the window; leave it
    at its default, or set it to `ts[0]` as
    [`from_uniform`][jbubble.pulse.sampled.SampledPulse.from_uniform] does.

    The inherited `envelope`, by default
    [`SoftRectangularEnvelope`][jbubble.pulse.envelope.SoftRectangularEnvelope],
    gates the signal to that window. Its sigmoid ramps are centred on the
    window edges, so it halves the first and last samples. To keep them at
    full value, pass a
    [`RectangularEnvelope`][jbubble.pulse.envelope.RectangularEnvelope] as
    `envelope`.

    Parameters
    ----------
    ts : jax.Array, shape (N,)
        Absolute sample times [s]. Must be monotonically increasing.
    pressures : jax.Array, shape (N,)
        Pressure values [Pa] at each sample time.

    Raises
    ------
    ValueError
        If `initial_time` is neither `0.0` nor `ts[0]`. The check runs only
        when both values are concrete, not traced.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jbubble.pulse import SampledPulse
    >>> ts = jnp.linspace(5e-6, 15e-6, 1000)
    >>> pressures = 200e3 * jnp.sin(2 * jnp.pi * 1e6 * ts)
    >>> pulse = SampledPulse(ts=ts, pressures=pressures)
    >>> print(f"{pulse.t_start * 1e6:.1f} µs to {pulse.t_stop * 1e6:.1f} µs")
    5.0 µs to 15.0 µs
    """

    ts: jax.Array
    pressures: jax.Array

    def __check_init__(self) -> None:
        initial_time = _concrete(self.initial_time)
        ts = _concrete(self.ts)
        if initial_time is None or ts is None or ts.size == 0:
            return
        start, first = float(initial_time), float(ts[0])
        if start != 0.0 and not np.isclose(start, first, rtol=1e-6, atol=0.0):
            raise ValueError(
                "SampledPulse starts at its first sample time, ts[0] = "
                f"{first!r} s, so initial_time must be 0.0 or ts[0]; got "
                f"{start!r} s. To delay the waveform, shift ts instead."
            )

    @property
    def t_start(self) -> jax.Array:
        """First sample time, `ts[0]` [s], where the active window starts."""
        return self.ts[0]

    @property
    def duration(self) -> jax.Array:
        """Time from the first sample to the last, `ts[-1] - ts[0]` [s]."""
        return self.ts[-1] - self.ts[0]

    def _evaluate(self, t: jax.Array) -> jax.Array:
        return jnp.interp(t, self.ts, self.pressures)

    @staticmethod
    def from_uniform(
        pressures: jax.Array, dt: float, initial_time: float = 0.0
    ) -> SampledPulse:
        """Create a `SampledPulse` from uniformly spaced samples.

        Parameters
        ----------
        pressures : jax.Array, shape (N,)
            Pressure values [Pa].
        dt : float
            Time step between samples [s].
        initial_time : float
            Time of the first sample [s]. Default: `0.0`.

        Returns
        -------
        SampledPulse
            Pulse with `ts = initial_time + dt * arange(N)` and
            `initial_time` set to the time of the first sample.
        """
        ts = initial_time + jnp.arange(pressures.shape[0]) * dt
        return SampledPulse(ts=ts, pressures=pressures, initial_time=initial_time)
