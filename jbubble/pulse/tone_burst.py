"""Parametric tone-burst pulse — fixed frequency, shape, and envelope."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .base import Pulse
from .shapes import PulseShape


class ToneBurst(Pulse):
    """Tone burst: carrier waveform × envelope × pressure amplitude.

    A ``ToneBurst`` is the standard parametric pulse used in most ultrasound
    simulations.  It combines a periodic :class:`PulseShape` (e.g. ``Sine``)
    with an :class:`Envelope` (e.g. ``HannEnvelope``) and a peak pressure.

    The ``initial_time`` and ``envelope`` fields are inherited from
    :class:`Pulse` and can be set as keyword arguments.

    Parameters
    ----------
    freq : float
        Carrier frequency [Hz].
    pressure : float
        Peak pressure amplitude [Pa].
    shape : PulseShape
        Waveform shape (``Sine``, ``Sawtooth``, …).
    phase : float
        Carrier phase offset [rad].  Default: 0.
    cycle_num : float
        Number of carrier cycles in the burst.  Default: 4.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jbubble.pulse import ToneBurst
    >>> from jbubble.pulse.shapes import Sine
    >>> pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)
    >>> float(pulse(jnp.array(0.0))) < 0.5  # sigmoid is ~0.5 at t=0
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
