"""jbubble: differentiable microbubble dynamics primitives."""

from .units import Units
from .bubble import Bubble
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
)
from .solver import bubble_equation, solve_bubble, SaveSpec
from .simulation import (
    SimulationResult,
    run_simulation,
    compute_radius_metrics,
)
from .visuals import arrays_from_result, line_figure, line_trace, PlotArrays

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
    "bubble_equation",
    "solve_bubble",
    "SaveSpec",
    "SimulationResult",
    "build_pulse",
    "default_pulse",
    "run_simulation",
    "compute_radius_metrics",
    "PlotArrays",
    "arrays_from_result",
    "line_trace",
    "line_figure",
]
