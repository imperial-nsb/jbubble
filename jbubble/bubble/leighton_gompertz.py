"""
Leighton model for a bubble confined in a rigid-walled tube.
"""

import jax
import jax.numpy as jnp
from typing import Any

from ..units import Units
from .base import Bubble
from ._gompertz import gompertz_surface_tension


class LeightonGompertz(Bubble):
    """
    Leighton Model:
    A lipid-coated microbubble confined in a rigid-walled tube and following a differentiable surface tension law (Gompertz)

    Models a bubble confined within a rigid-walled tube, incorporating:
    - Shell elasticity and viscosity (chi and kappa_s)
    - Smooth Gompertz surface tension
    - Polytropic gas law
    - Tube wall inertia and geometry effects
    - Liquid compressibility correction
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
    tube_radius: float
    tube_length: float

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
        tube_radius: float = 10.0e-6,
        tube_length: float = 100.0e-6,
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
        self.tube_radius = tube_radius
        self.tube_length = tube_length

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

    def get_scaled(self, units: Units) -> "LeightonGompertz":
        return LeightonGompertz(
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
            tube_radius=self.tube_radius / units.L_scale,
            tube_length=self.tube_length / units.L_scale,
        )

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        R, R_dot = state

        sigma = self.surface_tension(R)

        tuberad = self.tube_radius     # Γ1 (tube radius)
        zeta = self.tube_length / 2.0  # ζ1 (half-length)

        # Geometry factor per Leighton:
        # alpha = (ζ1/Γ1) * (1 + 8Γ1/(3πζ1)) - 1
        alpha = (zeta / tuberad) * (1.0 + (8.0 * tuberad) / (3.0 * jnp.pi * zeta)) - 1.0
        beta  = 2.0 * alpha  # curly-brace factor in the paper

        # Driving & gas term with small-order compressibility
        P_drive = pulse(t)
        damping_term = 1.0 - (3.0 * self.gamma * R_dot) / self.c_L
        P_gas = self.P_gas0 * (self.R0 / R) ** (3.0 * self.gamma) * damping_term

        # Dissipative and capillary terms
        P_surface_visc = 4.0 * self.kappa_s * R_dot / (R**2)
        P_liq_visc     = 4.0 * self.mu_L * R_dot / R
        P_Laplace      = 2.0 * sigma / R

        # Net forcing (right-hand side)
        rhs = (P_gas - P_Laplace - P_liq_visc - P_surface_visc - P_drive - self.P_amb) / self.rho_L

        # Leighton tube-inertia multipliers:
        # R*R̈ * [1 + (R/Γ1) β] + (3/2) Ṙ² * [1 + (4R/(3Γ1)) β] = rhs
        denom = R * (1.0 + (R / tuberad) * beta)
        inert = 1.5 * (R_dot ** 2) * (1.0 + (4.0 * R) / (3.0 * tuberad) * beta)

        R_ddot = (rhs - inert) / denom
        return jnp.stack([R_dot, R_ddot])
