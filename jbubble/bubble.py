"""Bubble material model definitions."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .units import Units


class Bubble(eqx.Module):
    """Encodes the Marmottant shell model parameters for a single bubble."""

    R0: float  # Initial radius [m]
    R_buckle: float  # Buckling radius [m]
    gamma: float  # Polytropic index of the gas
    chi: float  # Shell elasticity [N/m]
    mu_L: float  # Liquid viscosity [Pa.s]
    kappa_s: float  # Shell viscosity [kg/s]
    rho_L: float  # Liquid density [kg/m^3]
    c_L: float  # Speed of sound in liquid [m/s]
    P_amb: float  # Ambient pressure [Pa]
    sigma_L: float  # Surface tension of the liquid [N/m]
    R_break: float  # Break-up radius [m]
    sigma_break: float  # Surface tension at break-up [N/m]
    sigma_R0: float  # Initial surface tension [N/m]
    vdw: float  # Van der Waals hard core radius [m]

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
        vdw_divisor: float = 5.61,
    ) -> None:
        self.R0 = R0
        self.R_buckle = R_buckle if R_buckle is not None else 0.99 * R0
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
            jnp.where(R >= self.R_break, self.sigma_break, sigma_elastic),
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

    def __str__(self) -> str:
        """Print the bubble properties with units."""
        return "\n".join([
            "** Bubble Properties **",
            f"Initial radius (R0): \t\t\t{self.R0:.2e} [m]",
            f"Buckling radius (R_buckle): \t\t{self.R_buckle:.2e} [m]",
            f"Polytropic index (gamma): \t\t{self.gamma:.2f}",
            f"Shell elasticity (chi): \t\t{self.chi:.2f} [N/m]",
            f"Liquid viscosity (mu_L): \t\t{self.mu_L:.2e} [Pa.s]",
            f"Shell viscosity (kappa_s): \t\t{self.kappa_s:.2e} [kg/s]",
            f"Liquid density (rho_L): \t\t{self.rho_L:.1f} [kg/m^3]",
            f"Speed of sound (c_L): \t\t\t{self.c_L:.1f} [m/s]",
            f"Ambient pressure (P_amb): \t\t{self.P_amb:.2e} [Pa]",
            f"Liquid surface tension (sigma_L): \t{self.sigma_L:.2e} [N/m]",
            f"Break-up radius (R_break): \t\t{self.R_break:.2e} [m]",
            f"Break-up surface tension (sigma_break): {self.sigma_break:.2e} [N/m]",
            f"Initial surface tension (sigma_R0): \t{self.sigma_R0:.2e} [N/m]",
            f"Van der Waals radius (vdw): \t\t{self.vdw:.2e} [m]",
        ])
