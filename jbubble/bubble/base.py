"""
Base class for bubble material models.
"""

import abc
import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..units import Units
from . import _defaults, _pressure

State = jax.Array

# ---------------------------------------------------------------------------
# Field-scaling registry
# ---------------------------------------------------------------------------
# Maps field names to their corresponding ``Units`` attribute.  Used by the
# default ``get_scaled()`` to non-dimensionalise constructor parameters.

_FIELD_SCALES: dict[str, str] = {
    # Lengths
    "R0": "L_scale",
    "d_s": "L_scale",
    "tube_radius": "L_scale",
    "tube_length": "L_scale",
    "vessel_radius": "L_scale",
    "vessel_d": "L_scale",
    "tissue_d": "L_scale",
    # Pressures / elastic moduli
    "P_amb": "P_scale",
    "G": "P_scale",
    "G_s": "P_scale",
    "vessel_E": "P_scale",
    # Surface tensions / shell elasticity
    "chi": "chi_scale",
    "sigma_L": "sigma_scale",
    "sigma_m": "sigma_scale",
    # Dynamic viscosities
    "mu_L": "mu_scale",
    "mu_m": "mu_scale",
    "mu_s": "mu_scale",
    # Shell surface-dilatational viscosity
    "kappa_s": "kappa_scale",
    # Densities
    "rho_L": "rho_scale",
    "vessel_rho": "rho_scale",
    "tissue_rho": "rho_scale",
    # Velocities
    "c_L": "vel_scale",
    # Dimensionless
    "gamma": "unit_scale",
    "vdw_divisor": "unit_scale",
    "R_buckle_ratio": "unit_scale",
}


class Bubble(eqx.Module, abc.ABC):
    """
    Abstract base class for bubble material models.

    Concrete subclasses must implement:
        - surface_tension(R)   — interface pressure law
        - bubble_equation(t, state, pulse) — ODE right-hand side

    Optional overrides:
        - get_scaled(units)    — default scales all dataclass fields
        - chi_R(R)             — defaults to (R/2) dσ/dR via autodiff
        - initial_state()      — defaults to [R0, 0]
        - rescale_state(state, units) — defaults to [L_scale, vel_scale] scaling

    ``Bubble`` itself cannot be instantiated; import it for type annotations.
    """

    R0: float
    P_amb: float = _defaults.P_ATM

    @abc.abstractmethod
    def surface_tension(self, R: jax.Array) -> jax.Array:
        """Return the interface surface tension at radius R."""
        ...

    @property
    def P_gas0(self) -> float:
        """Equilibrium gas pressure, using the surface tension evaluated at R0."""
        return _pressure.gas_pressure_equilibrium(
            self.P_amb, self.surface_tension(self.R0), self.R0
        )

    def chi_R(self, R: jax.Array) -> jax.Array:
        """
        Shell elasticity parameter χ(R) = (R/2) dσ/dR.

        Default implementation uses automatic differentiation of
        ``surface_tension``.  Override for closed-form expressions or
        to return zero (e.g. ``RayleighPlesset``).
        """
        d_sigma_dR = jax.vmap(jax.grad(lambda r: self.surface_tension(r)))(R)
        return 0.5 * R * d_sigma_dR

    def get_scaled(self, units: Units) -> "Bubble":
        """Return a dimensionless copy of this model scaled by *units*.

        Iterates over ``dataclasses.fields`` (i.e. the constructor
        parameters) and divides each by the appropriate unit scale.
        Properties and derived quantities are recomputed automatically
        by the constructor.
        """
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            kwargs[f.name] = val / getattr(units, _FIELD_SCALES[f.name])
        return type(self)(**kwargs)

    @abc.abstractmethod
    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        """ODE right-hand side: return d(state)/dt."""
        ...

    def initial_state(self) -> jax.Array:
        """Default initial state [R0, 0]."""
        return jnp.array([self.R0, 0.0])

    def rescale_state(self, state: jax.Array, units: Units) -> jax.Array:
        """Rescale the state variables back to physical units.
        Override this for models with different state definitions.
        """
        scale_factors = jnp.array([units.L_scale, units.vel_scale])
        return state * scale_factors


class GompertzBubble(Bubble, abc.ABC):
    """
    Intermediate base class for models using Gompertz surface tension.

    Subclasses must provide (as fields or properties): ``sigma_break``.
    This class provides the shared ``surface_tension()`` implementation.
    """

    chi: float = _defaults.CHI_LIPID
    gamma: float = _defaults.GAMMA_LIPID
    rho_L: float = _defaults.RHO_WATER
    R_buckle_ratio: float = _defaults.R_BUCKLE_RATIO

    @property
    def R_buckle(self) -> float:
        return self.R0 * self.R_buckle_ratio

    @property
    def sigma_R0(self) -> float:
        return self.chi * ((self.R0 / self.R_buckle) ** 2 - 1.0)

    @property
    def R_break(self) -> float:
        return _defaults.R_BREAK_RATIO * self.R0

    @property
    @abc.abstractmethod
    def sigma_break(self) -> float: ...

    def surface_tension(self, R: jax.Array) -> jax.Array:
        """Evaluate Gompertz surface tension at radius *R*."""
        from ._gompertz import gompertz_surface_tension

        return gompertz_surface_tension(
            R,
            R0=self.R0,
            R_buckle=self.R_buckle,
            chi=self.chi,
            sigma_break=self.sigma_break,
            sigma_R0=self.sigma_R0,
        )
