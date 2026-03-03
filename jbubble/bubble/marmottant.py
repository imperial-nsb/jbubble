"""
Marmottant shell model for encapsulated microbubbles.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import _defaults, _pressure
from .base import Bubble


class Marmottant(Bubble):
    """
    Marmottant shell model for encapsulated microbubbles.

    A lipid-coated microbubble in a Newtonian fluid following a
    discontinuous piecewise surface tension law.

    Surface tension follows three regimes:
        1) Buckled:     σ = 0
        2) Elastic:     σ = χ (R²/R_buckle² − 1)
        3) Ruptured:    σ = σ_L

    Includes:
        - Shell elasticity and shell viscosity
        - Van der Waals gas correction
        - Liquid compressibility correction
    """

    gamma: float = _defaults.GAMMA_LIPID
    chi: float = _defaults.CHI_LIPID
    mu_L: float = _defaults.MU_WATER
    kappa_s: float = _defaults.KAPPA_S_LIPID
    rho_L: float = _defaults.RHO_WATER
    c_L: float = _defaults.C_WATER
    sigma_L: float = _defaults.SIGMA_WATER
    vdw_divisor: float = _defaults.VDW_DIVISOR
    R_buckle_ratio: float = _defaults.R_BUCKLE_RATIO

    @property
    def R_buckle(self):
        return self.R0 * self.R_buckle_ratio

    @property
    def sigma_R0(self):
        return self.chi * ((self.R0 / self.R_buckle) ** 2 - 1.0)

    @property
    def sigma_break(self):
        return self.sigma_L

    @property
    def R_break(self):
        return jnp.sqrt((self.sigma_break / self.chi) + 1) * self.R_buckle

    @property
    def vdw(self):
        return self.R0 / self.vdw_divisor

    def surface_tension(self, R: jax.Array) -> jax.Array:
        sigma_elastic = self.chi * ((R**2 / self.R_buckle**2) - 1.0)

        return jnp.where(
            self.R_buckle >= R,
            0.0,
            jnp.where(self.R_break <= R, self.sigma_L, sigma_elastic),
        )

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
