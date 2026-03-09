"""
Bubble material model definitions for single-bubble dynamics.

Includes Rayleigh-Plesset, Marmottant, Gompertz-smoothed shell,
Kelvin-Voigt viscoelastic, and spherical-confinement models.

The modular composition-based API is available via the ``interfaces``
and ``components`` submodules::

    from jbubble.bubble.interfaces import GasModel, ShellModel, ...
    from jbubble.bubble.components import PolytropicGas, MarmottantShell, ...
"""

from .._old import _pressure
from .._old.church_gompertz import ChurchGompertz
from . import _defaults, _gompertz  # noqa: F401
from .base import Bubble, State
from .eom import KellerMiksis, ModifiedRayleighPlesset, RayleighPlesset
from .gas import PolytropicGas, VanDerWaalsGas
from .medium import KelvinVoigtMedium, NeoHookeanMedium
from .shell import LipidShell, NoShell, ThickShell
from .surface import ConstantSigma, GompertzSigma, MarmottantSigma

__all__ = [
    # --- Legacy monolithic models ---
    "Bubble",
    "GompertzBubble",
    "State",
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
    "RayleighPlesset",
    "ModifiedRayleighPlesset",
    "KellerMiksis",
]
