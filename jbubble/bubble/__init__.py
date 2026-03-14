"""
Modular bubble dynamics components.

Compose gas laws, shell coatings, surrounding-medium rheologies, and
equations of motion independently::

    from jbubble.bubble import (
        PolytropicGas, GompertzSurfaceTension, LipidShell,
        KelvinVoigtMedium, KellerMiksis, ConstantProperty,
    )

    R0 = 2.5e-6
    sigma = GompertzSurfaceTension.from_R0(R0=R0)
    shell  = LipidShell(
        sigma=sigma,
        kappa_s=ConstantProperty(val=2.4e-9),
    )
    gas    = PolytropicGas.from_equilibrium(
        R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0,
    )
    medium = KelvinVoigtMedium(R0=R0, G=0.0, mu=0.00089)
    eom    = KellerMiksis(
        gas=gas, shell=shell, medium=medium,
        R0=R0, P_amb=101325.0, rho_L=998.0, c_L=1481.0,
    )
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
from .properties import (
    ConstantProperty,
    GompertzSurfaceTension,
    MarmottantSurfaceTension,
    Property,
)
from .shell import LipidShell, NoShell, ShellModel, ThickShell
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
