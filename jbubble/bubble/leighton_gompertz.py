"""
Leighton model for a bubble confined in a rigid-walled tube.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import _defaults, _pressure
from .base import GompertzBubble


class LeightonGompertz(GompertzBubble):
    """
    Leighton model for a bubble confined in a rigid-walled tube with
    Gompertz surface tension.

    Models a bubble confined within a rigid-walled tube, incorporating:
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law
        - Tube wall inertia and geometry effects
        - Liquid compressibility correction
    """

    mu_L: float = _defaults.MU_WATER
    kappa_s: float = _defaults.KAPPA_S_LIPID
    c_L: float = _defaults.C_WATER
    sigma_L: float = _defaults.SIGMA_WATER
    tube_radius: float = 10.0e-6
    tube_length: float = 100.0e-6

    @property
    def sigma_break(self):
        return self.sigma_L

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        R, R_dot = state

        sigma = self.surface_tension(R)

        tuberad = self.tube_radius  # Γ1 (tube radius)
        zeta = self.tube_length / 2.0  # ζ1 (half-length)

        # Geometry factor per Leighton:
        # alpha = (ζ1/Γ1) * (1 + 8Γ1/(3πζ1)) - 1
        alpha = (zeta / tuberad) * (1.0 + (8.0 * tuberad) / (3.0 * jnp.pi * zeta)) - 1.0
        beta = 2.0 * alpha  # curly-brace factor in the paper

        # Driving & gas term with small-order compressibility
        P_drive = pulse(t)
        damping_term = 1.0 - (3.0 * self.gamma * R_dot) / self.c_L
        P_gas = (
            _pressure.gas_pressure(self.P_gas0, self.R0, R, self.gamma) * damping_term
        )

        # Dissipative and capillary terms
        P_surface_visc = _pressure.shell_viscous_pressure(self.kappa_s, R_dot, R)
        P_liq_visc = _pressure.viscous_pressure(self.mu_L, R_dot, R)
        P_Laplace = _pressure.laplace_pressure(sigma, R)

        # Net forcing (right-hand side)
        rhs = (
            P_gas - P_Laplace - P_liq_visc - P_surface_visc - P_drive - self.P_amb
        ) / self.rho_L

        # Leighton tube-inertia multipliers:
        # R*R̈ * [1 + (R/Γ1) β] + (3/2) Ṙ² * [1 + (4R/(3Γ1)) β] = rhs
        denom = R * (1.0 + (R / tuberad) * beta)
        inert = 1.5 * (R_dot**2) * (1.0 + (4.0 * R) / (3.0 * tuberad) * beta)

        R_ddot = (rhs - inert) / denom
        return jnp.stack([R_dot, R_ddot])
