"""
Keller-Miksis model with Gompertz surface tension law.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import _defaults, _pressure
from .base import GompertzBubble


class KellerMiksisGompertz(GompertzBubble):
    """
    Keller-Miksis model with Gompertz surface tension.

    A lipid-coated microbubble in a compressible liquid following a
    differentiable surface tension law (Gompertz).

    Includes:
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law
        - Liquid compressibility correction (Keller-Miksis formulation)
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
        R, R_dot = state

        sigma = self.surface_tension(R)
        dsigma_dR = jax.grad(lambda r: self.surface_tension(r))(R)
        P_drive = pulse(t)

        # Pressure components
        P_Laplace = _pressure.laplace_pressure(sigma, R)
        P_visc = _pressure.viscous_pressure(self.mu_L, R_dot, R)
        P_surf_visc = _pressure.shell_viscous_pressure(self.kappa_s, R_dot, R)

        dPg_dt = (
            (-3.0 * self.gamma)
            * self.P_gas0
            * (R_dot / R)
            * (self.R0 / R) ** (3 * self.gamma)
        )
        dPdrive_dt = jax.grad(lambda t_val: pulse(t_val))(t)
        dLaplace_dt = 2 * R_dot * (dsigma_dR / R - sigma / R**2)

        P_int = _pressure.gas_pressure(self.P_gas0, self.R0, R, self.gamma)

        A = (1 / self.rho_L) * (
            (1 + R_dot / self.c_L)
            * (P_int - P_drive - self.P_amb - P_Laplace - P_visc - P_surf_visc)
        )
        B = dPg_dt - dPdrive_dt - dLaplace_dt
        C = R * (1 - R_dot / self.c_L) + (1 / self.rho_L) * (R / self.c_L) * (
            4 * self.kappa_s / R**2 + 4 * self.mu_L / R
        )

        D = (
            A
            + (1 / self.rho_L)
            * (R / self.c_L)
            * (
                B
                + (8 * self.kappa_s * R_dot**2 / R**3)
                + (4 * self.mu_L * R_dot**2 / R**2)
            )
            - 1.5 * (1 - R_dot / (3 * self.c_L)) * R_dot**2
        )

        R_ddot = D / C
        return jnp.stack([R_dot, R_ddot])
