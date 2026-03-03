"""
Marmottant model with smooth Gompertz surface tension law.
"""

from . import _defaults
from .base import GompertzBubble
from .marmottant import _MarmottantEquation


class MarmottantGompertz(_MarmottantEquation, GompertzBubble):
    """
    Smooth Gompertz-based variant of the Marmottant shell model.

    The discontinuous piecewise surface tension law is replaced with
    a differentiable Gompertz function, making the model suitable for
    automatic differentiation, gradient-based optimisation, and inverse
    problems.

    Includes:
        - Shell elasticity and shell viscosity
        - Van der Waals gas correction
        - Liquid compressibility correction
    """

    mu_L: float = _defaults.MU_WATER
    kappa_s: float = _defaults.KAPPA_S_LIPID
    c_L: float = _defaults.C_WATER
    sigma_L: float = _defaults.SIGMA_WATER
    vdw_divisor: float = _defaults.VDW_DIVISOR

    @property
    def sigma_break(self):
        return self.sigma_L
