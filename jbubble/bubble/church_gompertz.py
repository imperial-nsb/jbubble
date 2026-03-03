"""
Church (1995) thick-shell model with Gompertz surface tension.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import _defaults, _pressure
from .base import GompertzBubble


class ChurchGompertz(GompertzBubble):
    """
    Church (1995) thick elastic shell model with Gompertz surface tension.

    Models a lipid-coated microbubble with a thick, viscoelastic shell
    following Church (1995). Shell mechanics are parameterised directly via
    shell thickness *d_s*, shear modulus *G_s*, and shell viscosity *mu_s*.

    The shell contributions to the pressure balance are::

        P_elastic = (4/3) G_s (d_s/R₀) [1 − (R₀/R)³]
        P_shell_visc = 4 μ_s d_s Ṙ / R²

    Includes:
        - Thick-shell elasticity (G_s, d_s)
        - Thick-shell viscosity (mu_s, d_s)
        - Smooth Gompertz surface tension (chi)
        - Liquid viscosity (mu_L)
        - Polytropic gas law
    """

    mu_L: float = _defaults.MU_WATER
    sigma_L: float = _defaults.SIGMA_WATER
    d_s: float = 4e-9
    G_s: float = 10e6
    mu_s: float = 0.5

    @property
    def sigma_break(self):
        return self.sigma_L

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:

        R, R_dot = state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas = _pressure.gas_pressure(self.P_gas0, self.R0, R, self.gamma)

        P_surf = _pressure.laplace_pressure(sigma, R)
        P_visc = _pressure.viscous_pressure(self.mu_L, R_dot, R)

        # Church (1995) thick-shell elastic and viscous contributions
        P_elastic = (
            (4.0 / 3.0) * self.G_s * (self.d_s / self.R0) * (1.0 - (self.R0 / R) ** 3)
        )
        P_shell_visc = 4.0 * self.mu_s * self.d_s * R_dot / R**2

        forces = (
            P_gas - P_surf - P_visc - P_elastic - P_shell_visc - P_drive - self.P_amb
        )

        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])
