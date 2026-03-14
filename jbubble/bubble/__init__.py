"""
Modular bubble dynamics components.

Compose gas laws, shell coatings, surrounding-medium rheologies, and
equations of motion independently
"""

from .base import BubbleState, ConfinedBubbleState, Property
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
from .shell import (
    GompertzSurfaceTension,
    LipidShell,
    MarmottantSurfaceTension,
    NoShell,
    ShellModel,
    ThickShell,
)

__all__ = [
    # State
    "BubbleState",
    "ConfinedBubbleState",
    # Property
    "Property",
    # Interfaces
    "GasModel",
    "ShellModel",
    "MediumModel",
    "EquationOfMotion",
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
