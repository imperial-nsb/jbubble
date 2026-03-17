"""Surrounding medium (fluid / tissue) models.

Computes the total inward viscous and elastic stresses exerted by the
surrounding medium on the bubble wall.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .property import Property, as_property
from .state import BubbleState


class MediumModel(eqx.Module, abc.ABC):
    """Surrounding medium (fluid / tissue) model.

    Computes the total inward viscous and elastic stresses exerted by the
    surrounding medium on the bubble wall.  For a Newtonian liquid this is
    just 4 mu Rdot / R.  Viscoelastic media (Kelvin-Voigt, Neo-Hookean, etc.)
    add elastic restoring forces.

    Subclasses must implement separate methods for the viscous and elastic
    contributions, which are then summed in the default ``__call__`` to get
    the total medium pressure p_medium(state).

    A plain float is accepted for ``mu`` and auto-converted to a
    ``Property`` in ``__post_init__``.

    Fields
    ------
    mu : float or Property
        Dynamic viscosity  [Pa s].
    """

    mu: Property = eqx.field(converter=as_property)

    @abc.abstractmethod
    def p_viscous(self, state: BubbleState) -> jax.Array:
        """Viscous contribution to the medium pressure."""
        ...

    @abc.abstractmethod
    def p_elastic(self, state: BubbleState) -> jax.Array:
        """Elastic contribution to the medium pressure."""
        ...

    def __call__(self, state: BubbleState) -> jax.Array:
        """Compute total medium pressure p_medium(state).

        Parameters
        ----------
        state : BubbleState
            Current bubble state.

        Returns
        -------
        scalar
            Total inward medium pressure (viscous + elastic).
        """
        return self.p_viscous(state) + self.p_elastic(state)


class NewtonianMedium(MediumModel):
    """Newtonian liquid medium.

    p_medium = 4 mu Rdot / R

    Fields
    ------
    mu : float or Property
        Dynamic viscosity  [Pa s].
    """

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu(state) * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0


class KelvinVoigtMedium(MediumModel):
    """Kelvin-Voigt viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R / R0)^3 - 1)

    Fields
    ------
    mu : float or Property
        Dynamic viscosity  [Pa s].
    R0 : float
        Equilibrium bubble radius  [m].
    G : float or Property
        Shear modulus  [Pa].  May be state-dependent (e.g. strain-stiffening).
    """

    G: Property = eqx.field(converter=as_property)

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu(state) * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return (4.0 / 3.0) * self.G(state) * ((state.R / state.R0) ** 3 - 1.0)


class NeoHookeanMedium(MediumModel):
    """Neo-Hookean finite-strain viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R0/R)^3 - (R/R0)^3)

    Correctly captures finite-strain behaviour at large oscillation
    amplitudes where the Kelvin-Voigt linear approximation breaks down.

    Fields
    ------
    mu : float or Property
        Dynamic viscosity  [Pa s].  A plain float is auto-converted.
    R0 : float
        Equilibrium bubble radius  [m].
    G : float or Property
        Shear modulus  [Pa].  A plain float is auto-converted.  May be
        state-dependent (e.g. strain-stiffening).
    """

    G: Property = eqx.field(converter=as_property)

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu(state) * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return (
            (4.0 / 3.0)
            * self.G(state)
            * ((state.R0 / state.R) ** 3 - (state.R / state.R0) ** 3)
        )


class PowerLawMedium(MediumModel):
    """Power-law (generalised Newtonian) surrounding medium.

    The effective viscosity depends on the shear rate at the bubble wall:

        η_eff = mu · |2 Ṙ/R|^(n-1)

    Integrating the resulting stress field over the incompressible flow
    surrounding the bubble (Ting 1975) gives the viscous pressure:

        p_visc = (4 mu / n) · (2|Ṙ/R|)^(n-1) · Ṙ/R

    The factor 1/n arises from the spatial variation of η with radius.
    For n = 1 this reduces exactly to the Newtonian result 4 mu Ṙ/R.

    Special cases:

        n < 1   shear-thinning  (biological tissue, blood, polymer gels)
        n = 1   Newtonian (recovers ``NewtonianMedium`` with viscosity mu)
        n > 1   shear-thickening

    Note: power-law fluids have unbounded viscosity at zero shear rate
    when n < 1.  A small floor ``eps`` on |2 Ṙ/R| prevents numerical
    divergence near turnaround points without affecting dynamics.

    Fields
    ------
    mu : float or Property
        Consistency index K  [Pa·s^n].  Equals the dynamic viscosity for
        n = 1.
    n_exp : float or Property
        Power-law exponent  (dimensionless, positive).
    eps : float
        Floor applied to |2 Ṙ/R|  [s⁻¹].  Default 1e-6.
    """

    n_exp: Property = eqx.field(converter=as_property)
    eps: float = 1e-6

    def p_viscous(self, state: BubbleState) -> jax.Array:
        n_val = self.n_exp(state)
        K_val = self.mu(state)
        gamma_dot = jnp.maximum(2.0 * jnp.abs(state.R_dot / state.R), self.eps)
        return 4.0 * K_val / n_val * gamma_dot ** (n_val - 1.0) * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0
