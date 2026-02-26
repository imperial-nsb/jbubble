"""
Base class for bubble material models.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Any

from ..units import Units

State = jax.Array


class Bubble(eqx.Module, abc.ABC):
    """
    Abstract base class for bubble material models.

    Concrete subclasses must implement:
        - surface_tension(R)   — interface pressure law
        - get_scaled(units)    — return a dimensionless copy of the model
        - bubble_equation(t, state, pulse) — ODE right-hand side

    Optional overrides:
        - chi_R(R)             — defaults to (R/2) dσ/dR via autodiff
        - initial_state()      — defaults to [R0, 0]
        - rescale_state(state, units) — defaults to [L_scale, vel_scale] scaling

    ``Bubble`` itself cannot be instantiated; import it for type annotations.
    """

    @abc.abstractmethod
    def surface_tension(self, R: jax.Array) -> jax.Array:
        """Return the interface surface tension at radius R."""
        ...

    def chi_R(self, R: jax.Array) -> jax.Array:
        """
        Shell elasticity parameter χ(R) = (R/2) dσ/dR.

        Default implementation uses automatic differentiation of
        ``surface_tension``.  Override for closed-form expressions or
        to return zero (e.g. ``RayleighPlesset``).
        """
        d_sigma_dR = jax.vmap(jax.grad(lambda r: self.surface_tension(r)))(R)
        return 0.5 * R * d_sigma_dR

    @abc.abstractmethod
    def get_scaled(self, units: Units) -> "Bubble":
        """Return a dimensionless copy of this model scaled by *units*."""
        ...

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
