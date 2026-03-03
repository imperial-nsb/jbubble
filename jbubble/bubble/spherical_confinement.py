"""
Spherical confinement model with Gompertz surface tension.
"""

from typing import Any

import jax
import jax.numpy as jnp

from ..units import Units
from . import _defaults, _pressure
from .base import GompertzBubble


class SphericalConfinement(GompertzBubble):
    """
    Spherical confinement model with Gompertz surface tension law.

    Models a lipid-coated bubble confined within a thin, elastic
    spherical shell. Captures key qualitative effects of confinement
    on bubble dynamics, such as the presence of two normal modes.

    Includes:
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law
        - Confinement force based on linear elasticity of the vessel
        - Liquid compressibility correction
    """

    R0: float
    gamma: float = _defaults.GAMMA_LIPID
    chi: float = _defaults.CHI_LIPID
    mu_L: float = _defaults.MU_WATER
    kappa_s: float = _defaults.KAPPA_S_LIPID
    rho_L: float = _defaults.RHO_WATER
    c_L: float = _defaults.C_WATER
    P_amb: float = _defaults.P_ATM
    sigma_L: float = _defaults.SIGMA_WATER
    vessel_radius: float = 15.0e-6
    vessel_rho: float = 900.0
    vessel_E: float = 1.0e6
    vessel_d: float = 1.0e-6
    tissue_rho: float = 900.0
    tissue_d: float = 1.0e-6
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
        return _defaults.R_BREAK_RATIO * self.R0

    @property
    def P_gas0(self):
        return _pressure.gas_pressure_equilibrium(
            self.P_amb, self.sigma_R0, self.R0
        )

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        R, R_dot, a, a_dot = state
        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        rhoL = self.rho_L
        mu = self.mu_L
        ks = self.kappa_s
        P0 = self.P_amb
        a0 = self.vessel_radius
        Ev = self.vessel_E
        nu = 0.5  # nearly incompressible wall

        P_gas = (
            self.P_gas0
            * (self.R0 / R) ** (3.0 * self.gamma)
            * (1.0 - 3.0 * self.gamma * R_dot / self.c_L)
        )

        A = R**2
        B = -(a**2)
        C = rhoL * R**2 * (1.0 / R - 1.0 / a)
        D = self.vessel_rho * self.vessel_d + self.tissue_rho * self.tissue_d

        E = 2.0 * a * a_dot**2 - 2.0 * R * R_dot**2

        P_wall = Ev * (a - a0) / ((1.0 - nu**2) * a**2)

        F = (
            P_gas
            - 2.0 * R * R_dot * rhoL * (1.0 / R - 1.0 / a)
            - _pressure.laplace_pressure(sigma, R)
            - 4.0 * mu * (R_dot / R + a_dot / a)
            - _pressure.shell_viscous_pressure(ks, R_dot, R)
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
        scale_factors = jnp.array(
            [units.L_scale, units.vel_scale, units.L_scale, units.vel_scale]
        )
        return state * scale_factors
