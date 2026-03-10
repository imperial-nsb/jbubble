"""jbubble: differentiable microbubble dynamics primitives."""

from .bubble import (
    ConstantSigma,
    EquationOfMotion,
    GasModel,
    GompertzSigma,
    KellerMiksis,
    KelvinVoigtMedium,
    LeightonTube,
    LipidShell,
    MarmottantSigma,
    MediumModel,
    ModifiedRayleighPlesset,
    NeoHookeanMedium,
    NoShell,
    PolytropicGas,
    RayleighPlesset,
    ShellModel,
    SphericalConfinement,
    SurfaceTensionModel,
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
from .units import Units

__all__ = [
    # Units & solver
    "Units",
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
    # Bubble interfaces
    "GasModel",
    "SurfaceTensionModel",
    "ShellModel",
    "MediumModel",
    "EquationOfMotion",
    # Gas models
    "PolytropicGas",
    "VanDerWaalsGas",
    # Surface tension models
    "ConstantSigma",
    "MarmottantSigma",
    "GompertzSigma",
    # Shell models
    "NoShell",
    "LipidShell",
    "ThickShell",
    # Medium models
    "KelvinVoigtMedium",
    "NeoHookeanMedium",
    # Equations of motion
    "RayleighPlesset",
    "ModifiedRayleighPlesset",
    "KellerMiksis",
    "LeightonTube",
    "SphericalConfinement",
]
