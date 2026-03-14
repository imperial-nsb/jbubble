"""jbubble: differentiable microbubble dynamics primitives."""

import jax

jax.config.update("jax_enable_x64", True)

from .bubble import (
    BubbleState,
    ConfinedBubbleState,
    ConstantProperty,
    EquationOfMotion,
    GasModel,
    GompertzSurfaceTension,
    KellerMiksis,
    KelvinVoigtMedium,
    LeightonTube,
    LipidShell,
    MarmottantSurfaceTension,
    MediumModel,
    ModifiedRayleighPlesset,
    NeoHookeanMedium,
    NewtonianMedium,
    NoShell,
    PolytropicGas,
    Property,
    RayleighPlesset,
    ShellModel,
    SphericalConfinement,
    ThickShell,
    VanDerWaalsGas,
)
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
from .solver import SaveSpec, solve_eom

__all__ = [
    # Solver
    "SaveSpec",
    "solve_eom",
    # Pulse
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
    # Simulation
    "SimulationResult",
    "run_simulation",
    "compute_radius_metrics",
    "PlotArrays",
    "arrays_from_result",
    # State
    "BubbleState",
    "ConfinedBubbleState",
    # Bubble interfaces
    "GasModel",
    "Property",
    "ShellModel",
    "MediumModel",
    "EquationOfMotion",
    # Property models
    "ConstantProperty",
    "GompertzSurfaceTension",
    "MarmottantSurfaceTension",
    # Gas models
    "PolytropicGas",
    "VanDerWaalsGas",
    # Shell models
    "NoShell",
    "LipidShell",
    "ThickShell",
    # Medium models
    "NewtonianMedium",
    "KelvinVoigtMedium",
    "NeoHookeanMedium",
    # Equations of motion
    "RayleighPlesset",
    "ModifiedRayleighPlesset",
    "KellerMiksis",
    "LeightonTube",
    "SphericalConfinement",
]
