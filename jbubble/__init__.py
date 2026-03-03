"""jbubble: differentiable microbubble dynamics primitives."""

from .bubble import Marmottant as Bubble
from .pulse import Pulse
from .shapes import (
    Asymmetrical,
    NegativeQuadratic,
    PulseShape,
    Quadratic,
    Sawtooth,
    Sine,
    SlantedSine,
    Square,
    TimeDomainSawtooth,
    TimeDomainSquare,
    TimeDomainTriangle,
    Triangle,
)
from .simulation import (
    PlotArrays,
    SimulationResult,
    arrays_from_result,
    compute_radius_metrics,
    run_simulation,
)
from .solver import SaveSpec, solve_bubble
from .units import Units

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
