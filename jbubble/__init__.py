"""jbubble: differentiable microbubble dynamics primitives."""

import jax

jax.config.update("jax_enable_x64", True)

from .bubble import (
    BubbleState,
    ConfinedBubbleState,
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
from .pulse import (
    Asymmetrical,
    ChirpPulse,
    Envelope,
    HannEnvelope,
    NegativeQuadratic,
    NeuralPulse,
    Pulse,
    PulseShape,
    Quadratic,
    RectangularEnvelope,
    SampledPulse,
    Sawtooth,
    Scaled,
    Sine,
    SlantedSine,
    Square,
    Summed,
    TimeDomainSawtooth,
    TimeDomainSquare,
    TimeDomainTriangle,
    ToneBurst,
    Triangle,
    TukeyEnvelope,
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
    # Pulse (base)
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
    # Pulse composition
    "Scaled",
    "Summed",
    # Pulse shapes
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
