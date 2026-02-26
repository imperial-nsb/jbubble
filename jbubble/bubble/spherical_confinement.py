"""
Spherical confinement model with Gompertz surface tension.
"""

import jax
import jax.numpy as jnp
from typing import Any

from ..units import Units
from .base import Bubble
from ._gompertz import gompertz_surface_tension


class SphericalConfinement(Bubble):
    """
    Spherical confinement model with Gompertz surface tension law.

    Models a lipid-coated bubble confined within an thin, elastic and spherical shell.

    This model captures the key qualitative effects of confinement on bubble dynamics, such as the presence of two normal modes, while maintaining computational tractability.
    Includes:
    - Shell elasticity and viscosity (chi and kappa_s)
    - Smooth Gompertz surface tension
    - Polytropic gas law
    - Confinement force based on linear elasticity of the shell
    - Liquid compressibility correction.
    """

    R0: float
    R_buckle: float
    gamma: float
    chi: float
    mu_L: float
    kappa_s: float
    rho_L: float
    c_L: float
    P_amb: float
    sigma_L: float
    vessel_radius: float
    vessel_rho: float
    vessel_E: float
    vessel_d: float
    tissue_rho: float
    tissue_d: float

    R_break: float
    sigma_break: float
    sigma_R0: float
    P_gas0: float

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
        vessel_radius: float = 15.0e-6,
        vessel_rho: float = 900.0,
        vessel_E: float = 1.0e6,
        vessel_d: float = 1.0e-6,
        tissue_rho: float = 900.0,
        tissue_d: float = 1.0e-6,
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
        self.vessel_radius = vessel_radius
        self.vessel_rho = vessel_rho
        self.vessel_E = vessel_E
        self.vessel_d = vessel_d
        self.tissue_rho = tissue_rho
        self.tissue_d = tissue_d

        # Shell rupture
        self.R_break = 1.1 * R0
        self.sigma_break = sigma_L
        self.sigma_R0 = chi * ((R0**2 / self.R_buckle**2) - 1.0)

        # Precompute gas pressure at equilibrium
        self.P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0

    def surface_tension(self, R: jax.Array) -> jax.Array:
        return gompertz_surface_tension(
            R,
            R0=self.R0,
            R_buckle=self.R_buckle,
            chi=self.chi,
            sigma_break=self.sigma_break,
            sigma_R0=self.sigma_R0,
        )

    def get_scaled(self, units: Units) -> "SphericalConfinement":
        return SphericalConfinement(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,
            gamma=self.gamma,
            chi=self.chi * units.sigma_scale,
            mu_L=self.mu_L / units.mu_scale,
            kappa_s=self.kappa_s / (units.mu_scale * units.L_scale),
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale,
            vessel_radius=self.vessel_radius / units.L_scale,
            vessel_rho=self.vessel_rho / units.rho_scale,
            vessel_E=self.vessel_E / units.P_scale,
            vessel_d=self.vessel_d / units.L_scale,
            tissue_rho=self.tissue_rho / units.rho_scale,
            tissue_d=self.tissue_d / units.L_scale,
        )

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        R, R_dot, a, a_dot = state
        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        rhoL = self.rho_L
        mu   = self.mu_L
        ks   = self.kappa_s
        P0   = self.P_amb
        a0   = self.vessel_radius
        Ev   = self.vessel_E
        nu   = 0.5   # nearly incompressible wall

        P_gas = self.P_gas0 * (self.R0 / R) ** (3.0 * self.gamma) \
                * (1.0 - 3.0 * self.gamma * R_dot / self.c_L)

        A = R**2
        B = -a**2
        C = rhoL * R**2 * (1.0 / R - 1.0 / a)
        D = self.vessel_rho * self.vessel_d + self.tissue_rho * self.tissue_d

        E = 2.0 * a * a_dot**2 - 2.0 * R * R_dot**2

        P_wall = Ev * (a - a0) / ((1.0 - nu**2) * a**2)

        F = (
            P_gas
            - 2.0 * R * R_dot * rhoL * (1.0 / R - 1.0 / a)
            - 2.0 * sigma / R
            - 4.0 * mu * (R_dot / R + a_dot / a)
            - 4.0 * ks * R_dot / R**2
            - P_wall
            - P0
            - P_drive
        )

        Delta = A * D - B * C

        Delta = jnp.where(jnp.abs(Delta) < 1e-14, 1e-14, Delta)

        R_ddot = (E * D - B * F) / Delta
        a_ddot = (A * F - C * E) / Delta

        return jnp.stack([R_dot, R_ddot, a_dot, a_ddot])

    def initial_state(self) -> jax.Array:
        return jnp.array([self.R0, 0.0, self.vessel_radius, 0.0])

    def rescale_state(self, state: jax.Array, units: Units) -> jax.Array:
        scale_factors = jnp.array([
            units.L_scale, units.vel_scale, units.L_scale, units.vel_scale
        ])
        return state * scale_factors
