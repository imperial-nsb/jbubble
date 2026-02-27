"""
Marmottant shell model for encapsulated microbubbles.
"""

import jax
import jax.numpy as jnp
from typing import Any

from ..units import Units
from .base import Bubble


class Marmottant(Bubble):
    """
    Marmottant shell model for encapsulated microbubbles.
    A lipid-coated microbubble in a Newtonian fluid following a discontinuous piecewise surface tension law.

    Surface tension follows three regimes:
        1) Buckled:     σ = 0
        2) Elastic:     σ = χ (R²/R_buckle² − 1)
        3) Ruptured:    σ = σ_L

    Includes:
        - Shell elasticity and shell viscosity
        - Van der Waals gas correction
        - Liquid compressibility correction
    """

    R0: jax.Array
    R_buckle: float
    gamma: float
    chi: float
    mu_L: float
    kappa_s: float
    rho_L: float
    c_L: float
    P_amb: float
    sigma_L: float

    R_break: float
    sigma_break: float
    sigma_R0: float
    vdw: float

    def __init__(
        self,
        R0: float,
        R_buckle: float | None = None,
        gamma: float = 1.07,
        chi: float = 0.38,
        mu_L: float = 0.00089,
        kappa_s: float = 2.4e-9,
        rho_L: float = 1000.0,
        c_L: float = 1498.0,
        P_amb: float = 101.3e3,
        sigma_L: float = 72e-3,
        vdw_divisor: float = 5.61,
    ) -> None:

        self.R0 = R0
        self.R_buckle = 0.99 * self.R0 if R_buckle is None else R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_L = mu_L
        self.kappa_s = kappa_s
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L

        self.sigma_break = self.sigma_L
        self.R_break = jnp.sqrt((self.sigma_break / self.chi) + 1) * self.R_buckle
        self.sigma_R0 = self.chi * ((self.R0**2 / self.R_buckle**2) - 1.0)
        self.vdw = self.R0 / vdw_divisor

    def surface_tension(self, R: jax.Array) -> jax.Array:

        sigma_elastic = self.chi * ((R**2 / self.R_buckle**2) - 1.0)

        return jnp.where(
            R <= self.R_buckle,
            0.0,
            jnp.where(R >= self.R_break, self.sigma_L, sigma_elastic),
        )

    def get_scaled(self, units: Units) -> "Marmottant":
        return Marmottant(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,
            gamma=self.gamma,
            chi=self.chi / units.chi_scale,
            mu_L=self.mu_L / units.mu_scale,
            kappa_s=self.kappa_s / units.kappa_scale,
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale,
            vdw_divisor=5.61,  # Keep vdw scaling consistent with R0
        )

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:

        R, R_dot = state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0
        P_gas = P_gas0 * ((self.R0**3 - self.vdw**3) /
                          (R**3 - self.vdw**3)) ** self.gamma

        P_surf = 2.0 * sigma / R
        P_visc = 4.0 * self.mu_L * R_dot / R
        P_surf_visc = 4.0 * self.kappa_s * R_dot / R**2

        damping_term = 1.0 - (
            3.0 * self.gamma * R_dot * R**3
        ) / (self.c_L * (R**3 - self.vdw**3))

        forces = (P_gas * damping_term
                  - P_surf - P_visc - P_surf_visc
                  - P_drive - self.P_amb)

        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])
