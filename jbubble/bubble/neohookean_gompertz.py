"""
Neo-Hookean finite-strain viscoelastic model with Gompertz surface tension.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import _defaults, _pressure
from .base import GompertzBubble


class NeoHookeanGompertz(GompertzBubble):
    """
    Lipid-coated microbubble in a Neo-Hookean finite-strain medium with
    Gompertz shell surface tension.

    Identical to ``KelvinVoigtGompertz`` except the elastic medium stress
    uses the Neo-Hookean constitutive law rather than the linear
    Kelvin-Voigt formulation::

        P_elastic = (4/3) G [(R₀/R)³ − (R/R₀)³]

    This correctly captures finite-strain behaviour (large oscillation
    amplitudes) where the Kelvin-Voigt linear approximation breaks down.

    Includes:
        - Medium viscosity (μ_m) and finite-strain elasticity (G)
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law

    No liquid compressibility correction is included.
    """

    R0: float
    gamma: float = _defaults.GAMMA_LIPID
    chi: float = _defaults.CHI_LIPID
    mu_m: float = _defaults.MU_WATER
    kappa_s: float = _defaults.KAPPA_S_LIPID
    rho_L: float = _defaults.RHO_WATER
    P_amb: float = _defaults.P_ATM
    sigma_m: float = _defaults.SIGMA_WATER
    G: float = 0.0
    R_buckle_ratio: float = _defaults.R_BUCKLE_RATIO

    @property
    def R_buckle(self):
        return self.R0 * self.R_buckle_ratio

    @property
    def sigma_R0(self):
        return self.chi * ((self.R0 / self.R_buckle) ** 2 - 1.0)

    @property
    def sigma_break(self):
        return self.sigma_m

    @property
    def R_break(self):
        return _defaults.R_BREAK_RATIO * self.R0

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:

        R, R_dot = state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas0 = _pressure.gas_pressure_equilibrium(
            self.P_amb, self.sigma_R0, self.R0
        )
        P_gas = _pressure.gas_pressure(P_gas0, self.R0, R, self.gamma)

        P_surf = _pressure.laplace_pressure(sigma, R)
        P_visc = _pressure.viscous_pressure(self.mu_m, R_dot, R)
        P_surf_visc = _pressure.shell_viscous_pressure(self.kappa_s, R_dot, R)

        # Neo-Hookean finite-strain elastic term
        P_elastic = (4.0 / 3.0) * self.G * ((self.R0 / R) ** 3 - (R / self.R0) ** 3)

        forces = (
            P_gas - P_surf - P_visc - P_surf_visc - P_drive - self.P_amb - P_elastic
        )

        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])
