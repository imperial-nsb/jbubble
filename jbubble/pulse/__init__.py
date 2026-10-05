"""Acoustic driving pulses for bubble dynamics simulations.

This package provides a flexible, composable system for defining the
acoustic pressure waveform that drives a bubble. Every pulse is an
Equinox module and is fully JAX-differentiable.

Pulse types:

- [`ToneBurst`][jbubble.pulse.tone_burst.ToneBurst]: parametric
  carrier × envelope (the classic pulse).
- [`SampledPulse`][jbubble.pulse.sampled.SampledPulse]: interpolated from
  discrete data.
- [`ChirpPulse`][jbubble.pulse.chirp.ChirpPulse]: linear or exponential
  frequency sweep.
- [`NeuralPulse`][jbubble.pulse.neural.NeuralPulse]: waveform
  parameterised by a neural network.

Composition:

- [`Scaled`][jbubble.pulse.base.Scaled]: amplitude scaling.
- [`Summed`][jbubble.pulse.base.Summed]: additive superposition.
- [`Offset`][jbubble.pulse.base.Offset]: constant offset.
- `pulse.windowed(envelope)`: applies an envelope to any pulse.

Examples
--------
>>> from jbubble.pulse import ToneBurst
>>> from jbubble.pulse.shapes import Sine
>>> pulse = ToneBurst(freq=1e6, pressure=200e3, shape=Sine())
"""

from .base import (
    Offset,
    Pulse,
    Scaled,
    Summed,
)
from .chirp import ChirpPulse
from .envelope import (
    Envelope,
    HannEnvelope,
    RectangularEnvelope,
    SoftRectangularEnvelope,
    TukeyEnvelope,
)
from .neural import NeuralPulse
from .sampled import SampledPulse
from .tone_burst import ToneBurst

__all__ = [
    # Base
    "Pulse",
    # Envelopes
    "Envelope",
    "RectangularEnvelope",
    "SoftRectangularEnvelope",
    "HannEnvelope",
    "TukeyEnvelope",
    # Pulse types
    "ToneBurst",
    "SampledPulse",
    "ChirpPulse",
    "NeuralPulse",
    # Composition
    "Scaled",
    "Summed",
    "Offset",
]
