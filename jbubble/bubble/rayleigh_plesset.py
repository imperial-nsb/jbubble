"""
Rayleigh-Plesset model for an uncoated gas bubble.
"""

from typing import Any

import jax
import jax.numpy as jnp

from ..units import Units
from .base import Bubble, State


class RayleighPlesset(Bubble):
    """Rayleigh-Plesset model for an uncoated gas bubble.

    Classic model for a spherical gas bubble in an incompressible Newtonian
    liquid. Surface tension is constant (no shell effects). This is the
    simplest available model and is useful as a baseline or for uncoated
    bubbles (e.g. air cavitation).

    The governing equation (incompressible, no shell) is::

        R R̈ + (3/2) Ṙ² = [P_gas(R) − 2σ/R − 4μṘ/R − P_amb − P_drive(t)] / ρ

    Parameters
    ----------
    R0 : float
        Equilibrium bubble radius [m].
    gamma : float
        Polytropic exponent of the gas. Default 1.4 (adiabatic air).
    mu_L : float
        Dynamic viscosity of the surrounding liquid [Pa·s].
    rho_L : float
        Liquid density [kg/m³].
    c_L : float
        Speed of sound in the liquid [m/s] (used only in scaled output).
    P_amb : float
        Ambient (static) pressure [Pa].
    sigma_L : float
        Liquid–gas surface tension [N/m].
    """

    R0: jax.Array
    gamma: float
    mu_L: float
    rho_L: float
    c_L: float
    P_amb: float
    sigma_L: float
    sigma_R0: float

    def __init__(
        self,
        R0: float,
        gamma: float = 1.4,
        mu_L: float = 0.00089,
        rho_L: float = 1000.0,
        c_L: float = 1498.0,
        P_amb: float = 101.3e3,
        sigma_L: float = 72e-3,
    ) -> None:

        self.R0 = R0
        self.gamma = gamma
        self.mu_L = mu_L
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L

        self.sigma_R0 = self.sigma_L

    def surface_tension(self, R: jax.Array) -> jax.Array:
        return self.sigma_L * jnp.ones_like(R)

    def chi_R(self, R: jax.Array) -> jax.Array:
        return jnp.zeros_like(R)

    def get_scaled(self, units: Units) -> "RayleighPlesset":
        return RayleighPlesset(
            R0=self.R0 / units.L_scale,
            gamma=self.gamma,
            mu_L=self.mu_L / units.mu_scale,
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale,
        )

    def bubble_equation(self, t: Any, state: State, pulse) -> State:

        R, R_dot = state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0
        P_gas = P_gas0 * ((self.R0**3) / (R**3)) ** self.gamma

        P_surf = 2.0 * sigma / R
        P_visc = 4.0 * self.mu_L * R_dot / R

        forces = P_gas - P_surf - P_visc - P_drive - self.P_amb
        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])
