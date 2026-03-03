"""
Rayleigh-Plesset model for an uncoated gas bubble.
"""

from typing import Any

import jax
import jax.numpy as jnp

from . import _defaults, _pressure
from .base import Bubble, State


class RayleighPlesset(Bubble):
    """Rayleigh-Plesset model for an uncoated gas bubble.

    Classic model for a spherical gas bubble in an incompressible Newtonian
    liquid. Surface tension is constant (no shell effects). This is the
    simplest available model and is useful as a baseline or for uncoated
    bubbles (e.g. air cavitation).

    The governing equation (incompressible, no shell) is::

        R R̈ + (3/2) Ṙ² = [P_gas(R) − 2σ/R − 4μṘ/R − P_amb − P_drive(t)] / ρ
    """

    gamma: float = _defaults.GAMMA_AIR
    mu_L: float = _defaults.MU_WATER
    rho_L: float = _defaults.RHO_WATER
    c_L: float = _defaults.C_WATER
    sigma_L: float = _defaults.SIGMA_WATER

    def surface_tension(self, R: jax.Array) -> jax.Array:
        return self.sigma_L * jnp.ones_like(R)

    def chi_R(self, R: jax.Array) -> jax.Array:
        return jnp.zeros_like(R)

    def bubble_equation(self, t: Any, state: State, pulse) -> State:

        R, R_dot = state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas = _pressure.gas_pressure(self.P_gas0, self.R0, R, self.gamma)

        P_surf = _pressure.laplace_pressure(sigma, R)
        P_visc = _pressure.viscous_pressure(self.mu_L, R_dot, R)

        forces = P_gas - P_surf - P_visc - P_drive - self.P_amb
        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])
