"""
Modular bubble dynamics components.

Compose gas laws, shell coatings, surrounding-medium rheologies, and
equations of motion independently::

    from jbubble.bubble import (
        PolytropicGas, GompertzSigma, LipidShell, KelvinVoigtMedium, KellerMiksis,
    )

    R0 = 2.5e-6
    sigma = GompertzSigma.from_R0(R0=R0)
    shell  = LipidShell(sigma=sigma)
    gas    = PolytropicGas.from_equilibrium(
        R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0,
    )
    medium = KelvinVoigtMedium(R0=R0)
    eom    = KellerMiksis(
        gas=gas, shell=shell, medium=medium,
        R0=R0, P_amb=101325.0, rho_L=998.0, c_L=1481.0,
    )
"""

from .eom import (
    KellerMiksis,
    LeightonTube,
    ModifiedRayleighPlesset,
    RayleighPlesset,
    SphericalConfinement,
)
from .gas import PolytropicGas, VanDerWaalsGas
from .interfaces import (
    EquationOfMotion,
    GasModel,
    MediumModel,
    ShellModel,
    State,
    SurfaceTensionModel,
)
from .medium import (
    KelvinVoigtMedium,
    NeoHookeanMedium,
    NewtonianMedium,
)
from .shell import LipidShell, NoShell, ThickShell
from .surface import (
    ConstantSigma,
    GompertzSigma,
    MarmottantSigma,
    gompertz_surface_tension,
)

__all__ = [
    # Interfaces
    "GasModel",
    "SurfaceTensionModel",
    "ShellModel",
    "MediumModel",
    "EquationOfMotion",
    "State",
    # Gas models
    "PolytropicGas",
    "VanDerWaalsGas",
    # Surface tension models
    "ConstantSigma",
    "MarmottantSigma",
    "GompertzSigma",
    "gompertz_surface_tension",
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
