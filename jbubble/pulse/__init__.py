"""Acoustic driving pulses for bubble dynamics simulations.

This module provides a flexible, composable system for defining the
acoustic pressure waveform that drives a bubble.  Every pulse is an
Equinox module and is fully JAX-differentiable.

Quick start
-----------
>>> from jbubble.pulse import ToneBurst, HannEnvelope
>>> from jbubble.pulse.shapes import Sine
>>> pulse = ToneBurst(freq=1e6, pressure=200e3, shape=Sine(),
...                   envelope=HannEnvelope())

Pulse types
-----------
- :class:`ToneBurst` — parametric carrier × envelope (the classic pulse)
- :class:`SampledPulse` — interpolated from discrete data
- :class:`ChirpPulse` — linear or exponential frequency sweep
- :class:`NeuralPulse` — neural-network-parameterised waveform

Composition
-----------
- :class:`Scaled` — amplitude scaling
- :class:`Summed` — additive superposition
- ``pulse.windowed(envelope)`` — apply an envelope to any pulse
"""

from .base import (
    Envelope,
    HannEnvelope,
    Pulse,
    RectangularEnvelope,
    Scaled,
    Summed,
    TukeyEnvelope,
)
from .chirp import ChirpPulse
from .neural import NeuralPulse
from .sampled import SampledPulse
from .shapes import (
    Asymmetrical,
    FourierPulseShape,
    InvertedSawtooth,
    Mono95,
    Mono99,
    NegativeQuadratic,
    Pulse9,
    Pulse10,
    PulseShape,
    Quadratic,
    Rect25,
    Rect25NegPos,
    Rect75,
    Rect75NegPos,
    Rect95,
    Sawtooth,
    Sine,
    SlantedSine,
    Square,
    SquareNegPos,
    TimeDomainSawtooth,
    TimeDomainSquare,
    TimeDomainTriangle,
    Triangle,
)
from .tone_burst import ToneBurst

__all__ = [
    # Base
    "Pulse",
    "Envelope",
    "RectangularEnvelope",
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
    # Shapes
    "PulseShape",
    "FourierPulseShape",
    "Sine",
    "Sawtooth",
    "InvertedSawtooth",
    "Triangle",
    "Quadratic",
    "NegativeQuadratic",
    "Asymmetrical",
    "SlantedSine",
    "Square",
    "TimeDomainSquare",
    "TimeDomainSawtooth",
    "TimeDomainTriangle",
    "Pulse9",
    "Pulse10",
    "Rect75",
    "Rect25",
    "Mono99",
    "Mono95",
    "Rect75NegPos",
    "Rect25NegPos",
    "SquareNegPos",
    "Rect95",
]
