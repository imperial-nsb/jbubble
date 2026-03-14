"""
Modular bubble dynamics components.

Compose gas laws, shell coatings, surrounding-medium rheologies, and
equations of motion independently
"""

from .eom import (
    EquationOfMotion,
    KellerMiksis,
    LeightonTube,
    ModifiedRayleighPlesset,
    RayleighPlesset,
    SphericalConfinement,
)
from .gas import GasModel, PolytropicGas, VanDerWaalsGas
from .medium import (
    KelvinVoigtMedium,
    MediumModel,
    NeoHookeanMedium,
    NewtonianMedium,
)
from .properties import Property
from .shell import (
    GompertzSurfaceTension,
    LipidShell,
    MarmottantSurfaceTension,
    NoShell,
    ShellModel,
    ThickShell,
)
from .state import BubbleState, ConfinedBubbleState

__all__ = [
    # State
    "BubbleState",
    "ConfinedBubbleState",
    # Interfaces
    "GasModel",
    "Property",
    "ShellModel",
    "MediumModel",
    "EquationOfMotion",
    # Property models
    "Property",
    # Gas models
    "PolytropicGas",
    "VanDerWaalsGas",
    # Shell & surface tension models
    "NoShell",
    "LipidShell",
    "ThickShell",
    "GompertzSurfaceTension",
    "MarmottantSurfaceTension",
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
