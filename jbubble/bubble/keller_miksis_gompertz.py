"""
Keller-Miksis model with Gompertz surface tension law.
"""

import jax
import jax.numpy as jnp
from typing import Any

from ..units import Units
from .base import Bubble
from ._gompertz import gompertz_surface_tension


class KellerMiksisGompertz(Bubble):
    """
    Keller-Miksis-Gompertz Model:
    A lipid-coated microbubble in a non-newtonian fluid following a differentiable surface tension law (Gompertz)

    Includes:
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law
        - Liquid compressibility correction based on Keller-Miksis formulation
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

    def get_scaled(self, units: Units) -> "KellerMiksisGompertz":
        return KellerMiksisGompertz(
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
        )

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        R, R_dot = state

        sigma = self.surface_tension(R)
        dsigma_dR = jax.grad(lambda r: self.surface_tension(r))(R)
        P_drive = pulse(t)

        # Pressure components
        P_Laplace = 2.0 * sigma / R
        P_visc = 4.0 * self.mu_L * R_dot / R
        P_surf_visc = 4.0 * self.kappa_s * R_dot / R**2

        dPg_dt = (-3.0 * self.gamma) * self.P_gas0 * (R_dot / R) * (self.R0 / R) ** (3 * self.gamma)
        dPdrive_dt = jax.grad(lambda t_val: pulse(t_val))(t)
        dLaplace_dt = 2 * R_dot * (dsigma_dR / R - sigma / R**2)

        P_int = self.P_gas0 * (self.R0 / R) ** (3 * self.gamma)

        A = (1 / self.rho_L) * ((1 + R_dot / self.c_L) * (P_int - P_drive - self.P_amb - P_Laplace - P_visc - P_surf_visc))
        B = dPg_dt - dPdrive_dt - dLaplace_dt
        C = R * (1 - R_dot / self.c_L) + (1 / self.rho_L) * (R / self.c_L) * (4 * self.kappa_s / R**2 + 4 * self.mu_L / R)

        D = A + (1 / self.rho_L) * (R / self.c_L) * (B + (8 * self.kappa_s * R_dot**2 / R**3) + (4 * self.mu_L * R_dot**2 / R**2)) \
            - 1.5 * (1 - R_dot / (3 * self.c_L)) * R_dot**2

        R_ddot = D / C
        return jnp.stack([R_dot, R_ddot])
