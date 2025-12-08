"""Bubble material model definitions."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .units import Units


class Bubble(eqx.Module):
    """Encodes the Marmottant shell model parameters for a single bubble."""

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
    R_break: float
    sigma_break: float
    sigma_R0: float
    vdw: float

    def __init__(
        self,
        R0: float,
        R_buckle: float,
        gamma: float,
        chi: float,
        mu_L: float,
        kappa_s: float,
        rho_L: float,
        c_L: float,
        P_amb: float,
        sigma_L: float,
        vdw_divisor: float = 5.61,
    ) -> None:
        self.R0 = R0
        self.R_buckle = R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_L = mu_L
        self.kappa_s = kappa_s
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L
        self.R_break = 1.2 * self.R0
        self.sigma_break = ((self.R_break / self.R_buckle) ** 2 - 1.0) * self.chi
        self.sigma_R0 = self.chi * ((self.R0**2 / self.R_buckle**2) - 1.0)
        self.vdw = self.R0 / vdw_divisor

    def surface_tension(self, R: float) -> jax.Array:
        """Marmottant surface-tension law in a JAX-friendly form."""

        sigma_elastic = self.chi * ((R**2 / self.R_buckle**2) - 1.0)
        return jnp.where(
            R <= self.R_buckle,
            0.0,
            jnp.where(R >= self.R_break, self.sigma_L, sigma_elastic),
        )

    def get_scaled(self, units: Units) -> "Bubble":
        """Return a copy of the bubble scaled into dimensionless units."""

        return Bubble(
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
