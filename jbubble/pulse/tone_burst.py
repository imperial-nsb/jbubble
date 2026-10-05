"""Parametric tone-burst pulse with a fixed frequency, shape, and envelope."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .base import Pulse
from .shapes import PulseShape

__all__ = ["ToneBurst"]


class ToneBurst(Pulse):
    r"""Tone burst: carrier waveform × envelope × pressure amplitude.

    A `ToneBurst` is the standard parametric pulse in most ultrasound
    simulations. It combines a periodic
    [`PulseShape`][jbubble.pulse.shapes.PulseShape], such as
    [`Sine`][jbubble.pulse.shapes.Sine], with an
    [`Envelope`][jbubble.pulse.envelope.Envelope], such as
    [`HannEnvelope`][jbubble.pulse.envelope.HannEnvelope], and a peak
    pressure:

    $$
    p(t) = P\, s(t - t_0; f, \phi)\, w(t - t_0, T), \qquad T = \frac{N}{f},
    $$

    where $P$ is `pressure`, $s$ is `shape`, $f$ is `freq`, $\phi$ is
    `phase`, $w$ is `envelope`, $t_0$ is `initial_time`, and $N$ is
    `cycle_num`.

    `ToneBurst` inherits the `initial_time` and `envelope` fields from
    [`Pulse`][jbubble.pulse.base.Pulse]; set them as keyword arguments.

    Parameters
    ----------
    freq : float
        Carrier frequency [Hz].
    pressure : float
        Peak pressure amplitude [Pa].
    shape : PulseShape
        Waveform shape, such as `Sine` or `Sawtooth`.
    phase : float
        Carrier phase offset [rad]. Default: `0.0`.
    cycle_num : float
        Number of carrier cycles in the burst. Default: `4.0`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jbubble.pulse import ToneBurst
    >>> from jbubble.pulse.shapes import Sine
    >>> pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)
    >>> float(pulse(jnp.array(0.0))) < 0.5  # the soft envelope is ~0.5 at t=0
    True
    """

    freq: ArrayLike
    pressure: ArrayLike
    shape: PulseShape
    phase: ArrayLike = 0.0
    cycle_num: ArrayLike = 4.0

    @property
    def duration(self) -> float | jax.Array:
        return jnp.asarray(self.cycle_num) / jnp.asarray(self.freq)

    def _evaluate(self, t: jax.Array) -> jax.Array:
        return self.shape(t, self.freq, self.phase, self.initial_time) * self.pressure
