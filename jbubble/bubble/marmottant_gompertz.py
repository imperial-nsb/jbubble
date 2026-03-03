"""
Marmottant model with smooth Gompertz surface tension law.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import _defaults, _pressure
from .base import GompertzBubble


class MarmottantGompertz(GompertzBubble):
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

    @property
    def vdw(self):
        return self.R0 / self.vdw_divisor

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:

        R, R_dot = state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas = (
            self.P_gas0
            * ((self.R0**3 - self.vdw**3) / (R**3 - self.vdw**3)) ** self.gamma
        )

        P_surf = _pressure.laplace_pressure(sigma, R)
        P_visc = _pressure.viscous_pressure(self.mu_L, R_dot, R)
        P_surf_visc = _pressure.shell_viscous_pressure(self.kappa_s, R_dot, R)

        damping_term = 1.0 - (3.0 * self.gamma * R_dot * R**3) / (
            self.c_L * (R**3 - self.vdw**3)
        )

        forces = (
            P_gas * damping_term
            - P_surf
            - P_visc
            - P_surf_visc
            - P_drive
            - self.P_amb
        )

        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])
