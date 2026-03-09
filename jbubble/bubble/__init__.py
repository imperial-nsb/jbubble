"""
Bubble material model definitions for single-bubble dynamics.

Includes Rayleigh-Plesset, Marmottant, Gompertz-smoothed shell,
Kelvin-Voigt viscoelastic, and spherical-confinement models.

The modular composition-based API is available via the ``interfaces``
and ``components`` submodules::

    from jbubble.bubble.interfaces import GasModel, ShellModel, ...
    from jbubble.bubble.components import PolytropicGas, MarmottantShell, ...
"""

from . import _defaults, _gompertz, _pressure  # noqa: F401
from .base import Bubble, GompertzBubble, State
from .church_gompertz import ChurchGompertz
from .components import (
    ConstantSigma,
    GompertzSigma,
    KellerMiksis,
    KelvinVoigtMedium,
    LipidShell,
    MarmottantSigma,
    ModifiedRayleighPlesset,
    NeoHookeanMedium,
    NoShell,
    PolytropicGas,
    ThickShell,
    VanDerWaalsGas,
)
from .components import RayleighPlesset as RayleighPlessetEoM
from .interfaces import (
    EquationOfMotion,
    GasModel,
    MediumModel,
    ShellModel,
    SurfaceTensionModel,
)
from .keller_miksis_gompertz import KellerMiksisGompertz
from .kelvin_voigt_gompertz import KelvinVoigtGompertz
from .leighton_gompertz import LeightonGompertz
from .marmottant import Marmottant
from .marmottant_gompertz import MarmottantGompertz
from .neohookean_gompertz import NeoHookeanGompertz
from .rayleigh_plesset import RayleighPlesset
from .spherical_confinement import SphericalConfinement

__all__ = [
    # --- Legacy monolithic models ---
    "Bubble",
    "GompertzBubble",
    "State",
    "RayleighPlesset",
    "Marmottant",
    "MarmottantGompertz",
    "KelvinVoigtGompertz",
    "NeoHookeanGompertz",
    "KellerMiksisGompertz",
    "ChurchGompertz",
    "LeightonGompertz",
    "SphericalConfinement",
    # --- Modular interfaces ---
    "SurfaceTensionModel",
    "GasModel",
    "ShellModel",
    "MediumModel",
    "EquationOfMotion",
    # --- Modular components ---
    "PolytropicGas",
    "VanDerWaalsGas",
    "ConstantSigma",
    "MarmottantSigma",
    "GompertzSigma",
    "NoShell",
    "LipidShell",
    "ThickShell",
    "KelvinVoigtMedium",
    "NeoHookeanMedium",
    "RayleighPlessetEoM",
    "ModifiedRayleighPlesset",
    "KellerMiksis",
]
