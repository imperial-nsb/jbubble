"""
Kelvin-Voigt viscoelastic medium model with Gompertz surface tension.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import _defaults, _pressure
from .base import GompertzBubble


class KelvinVoigtGompertz(GompertzBubble):
    """
    Lipid-coated microbubble in a Kelvin-Voigt viscoelastic medium with
    Gompertz shell surface tension.

    The surrounding material is modeled as a Kelvin-Voigt solid::

        σ_medium = G · strain + μ_m · strain_rate

    Includes:
        - Medium viscosity (μ_m) and linear elastic modulus (G)
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law

    No liquid compressibility correction is included.
    """

    mu_m: float = _defaults.MU_WATER
    kappa_s: float = _defaults.KAPPA_S_LIPID
    sigma_m: float = _defaults.SIGMA_WATER
    G: float = 0.0

    @property
    def sigma_break(self):
        return self.sigma_m

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:

        R, R_dot = state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas = _pressure.gas_pressure(self.P_gas0, self.R0, R, self.gamma)

        P_surf = _pressure.laplace_pressure(sigma, R)
        P_visc = _pressure.viscous_pressure(self.mu_m, R_dot, R)
        P_surf_visc = _pressure.shell_viscous_pressure(self.kappa_s, R_dot, R)

        P_elastic = (4.0 / 3.0) * self.G * ((R**3 - self.R0**3) / self.R0**3)

        forces = (
            P_gas - P_surf - P_visc - P_surf_visc - P_drive - self.P_amb - P_elastic
        )

        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])
