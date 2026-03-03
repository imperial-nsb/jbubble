"""
Kelvin-Voigt viscoelastic medium model with Gompertz surface tension.
"""

from typing import Any

import jax
import jax.numpy as jnp

from ..units import Units
from ._gompertz import gompertz_surface_tension
from .base import Bubble


class KelvinVoigtGompertz(Bubble):
    """
    Lipid-coated microbubble in a Kelvin-Voigt viscoelastic medium with
    Gompertz shell surface tension.

    The surrounding material is modeled as a Kelvin-Voigt solid:
        σ_medium = G * strain + μ_m * strain_rate

    Includes:
        - Medium viscosity (μ_m)
        - Medium elastic modulus (G)
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law

    No liquid compressibility correction is included.
    """

    R0: jax.Array
    R_buckle: float
    gamma: float
    chi: float
    mu_m: float
    kappa_s: float
    rho_L: float
    P_amb: float
    sigma_m: float
    G: float

    R_break: float
    sigma_break: float
    sigma_R0: float

    def __init__(
        self,
        R0: float,
        R_buckle: float | None = None,
        gamma: float = 1.07,
        chi: float = 0.38,
        mu_m: float = 0.00089,
        kappa_s: float = 2.4e-9,
        rho_L: float = 1000.0,
        P_amb: float = 101.3e3,
        sigma_m: float = 72e-3,
        G: float = 0.0,
    ) -> None:
        """Initialise a KelvinVoigtGompertz bubble model.

        Parameters
        ----------
        R0 : float
            Equilibrium bubble radius [m].
        R_buckle : float, optional
            Shell buckling radius [m].  Defaults to ``0.99 * R0``.
        gamma : float
            Polytropic exponent of the enclosed gas.  Typical lipid shell:
            1.07; air: 1.4.
        chi : float
            Shell elasticity modulus [N/m].  Typical lipid shell: 0.2–0.6 N/m.
        mu_m : float
            Dynamic viscosity of the surrounding medium [Pa·s].
        kappa_s : float
            Shell surface dilatational viscosity [N·s/m].
        rho_L : float
            Medium density [kg/m³].
        P_amb : float
            Ambient pressure [Pa].
        sigma_m : float
            Surface tension of the bare medium–gas interface [N/m].
        G : float
            Shear modulus of the viscoelastic medium [Pa].  Set to ``0`` for
            a purely viscous liquid.
        """
        self.R0 = R0
        self.R_buckle = 0.99 * self.R0 if R_buckle is None else R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_m = mu_m
        self.kappa_s = kappa_s
        self.rho_L = rho_L
        self.P_amb = P_amb
        self.sigma_m = sigma_m
        self.G = G

        self.R_break = 1.1 * self.R0
        self.sigma_break = self.sigma_m
        self.sigma_R0 = self.chi * ((self.R0**2 / self.R_buckle**2) - 1.0)

    def surface_tension(self, R: jax.Array) -> jax.Array:
        return gompertz_surface_tension(
            R,
            R0=self.R0,
            R_buckle=self.R_buckle,
            chi=self.chi,
            sigma_break=self.sigma_break,
            sigma_R0=self.sigma_R0,
        )

    def get_scaled(self, units: Units) -> "KelvinVoigtGompertz":
        return KelvinVoigtGompertz(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,
            gamma=self.gamma,
            chi=self.chi / units.chi_scale,
            mu_m=self.mu_m / units.mu_scale,
            kappa_s=self.kappa_s / units.kappa_scale,
            rho_L=self.rho_L / units.rho_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_m=self.sigma_m / units.sigma_scale,
            G=self.G / units.P_scale,
        )

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:

        R, R_dot = state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0
        P_gas = P_gas0 * (self.R0 / R) ** (3 * self.gamma)

        P_surf = 2.0 * sigma / R
        P_visc = 4.0 * self.mu_m * R_dot / R
        P_surf_visc = 4.0 * self.kappa_s * R_dot / R**2

        P_elastic = (4.0 / 3.0) * self.G * ((R**3 - self.R0**3) / self.R0**3)

        forces = (
            P_gas - P_surf - P_visc - P_surf_visc - P_drive - self.P_amb - P_elastic
        )

        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])
