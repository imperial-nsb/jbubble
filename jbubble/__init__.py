"""jbubble: differentiable microbubble dynamics primitives."""

from .units import Units
from .bubble import RayleighPlesset as Bubble
from .pulse import Pulse
from .shapes import (
    PulseShape,
    Sine,
    Sawtooth,
    Triangle,
    Quadratic,
    NegativeQuadratic,
    Asymmetrical,
    SlantedSine,
    Square,
    TimeDomainSquare,
    TimeDomainSawtooth,
    TimeDomainTriangle,
)
from .solver import solve_bubble, SaveSpec
from .simulation import (
    SimulationResult,
    run_simulation,
    compute_radius_metrics,
    arrays_from_result,
    PlotArrays,
)

__all__ = [
    "Units",
    "Bubble",
    "Pulse",
    "PulseShape",
    "Sine",
    "Sawtooth",
    "Triangle",
    "Quadratic",
    "NegativeQuadratic",
    "Asymmetrical",
    "SlantedSine",
    "Square",
    "TimeDomainSquare",
    "TimeDomainSawtooth",
    "TimeDomainTriangle",
    "bubble_equation",
    "solve_bubble",
    "SaveSpec",
    "SimulationResult",
    "run_simulation",
    "compute_radius_metrics",
    "PlotArrays",
    "arrays_from_result",
]
