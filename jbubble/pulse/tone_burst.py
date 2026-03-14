"""Parametric tone-burst pulse — fixed frequency, shape, and envelope."""

import equinox as eqx
import jax

from .base import Envelope, Pulse, RectangularEnvelope
from .shapes import PulseShape


class ToneBurst(Pulse):
    """Tone burst: carrier waveform × envelope × pressure amplitude.

    A ``ToneBurst`` is the standard parametric pulse used in most ultrasound
    simulations.  It combines a periodic :class:`PulseShape` (e.g. ``Sine``)
    with an :class:`Envelope` (e.g. ``HannEnvelope``) and a peak pressure.

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
    initial_time : float
        Time at which the pulse begins [s].  Default: 0.
    cycle_num : float
        Number of carrier cycles in the burst.  Default: 4.
    envelope : Envelope
        Window applied to the burst.  Default: ``RectangularEnvelope()``
        (hard on/off gating).  Use ``HannEnvelope()`` for smooth transitions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jbubble.pulse import ToneBurst, HannEnvelope
    >>> from jbubble.pulse.shapes import Sine
    >>> pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(),
    ...                   cycle_num=5, envelope=HannEnvelope())
    >>> float(pulse(jnp.array(0.0)))
    0.0
    """

    freq: float
    pressure: float
    shape: PulseShape
    phase: float = 0.0
    initial_time: float = 0.0
    cycle_num: float = 4.0
    envelope: Envelope = eqx.field(default_factory=RectangularEnvelope)

    @property
    def duration(self) -> float:
        return self.cycle_num / self.freq

    @property
    def t_end(self) -> float:
        return self.initial_time + 2.0 * self.duration

    def __call__(self, t: jax.Array) -> jax.Array:
        tau = t - self.initial_time
        window = self.envelope(tau, self.duration)
        val = self.shape(t, self.freq, self.phase, self.initial_time)
        return val * self.pressure * window
