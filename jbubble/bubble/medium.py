"""Surrounding medium (fluid / tissue) models.

Computes the total inward viscous and elastic stresses exerted by the
surrounding medium on the bubble wall.
"""

import abc

import equinox as eqx
import jax

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
    """

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
    mu : float
        Dynamic viscosity  [Pa s].
    """

    mu: float

    @classmethod
    def water(cls, mu: float = 0.00089) -> "NewtonianMedium":
        """Construct with water viscosity (default 0.89 mPa s)."""
        return cls(mu=mu)

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0


class KelvinVoigtMedium(MediumModel):
    """Kelvin-Voigt viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R / R0)^3 - 1)

    Fields
    ------
    R0 : float
        Equilibrium bubble radius  [m].
    G : float
        Shear modulus  [Pa].
    mu : float
        Dynamic viscosity  [Pa s].
    """

    R0: float
    G: float
    mu: float

    @classmethod
    def water(
        cls,
        R0: float,
        G: float,
        mu: float = 0.00089,
    ) -> "KelvinVoigtMedium":
        """Construct with water-like viscosity.

        Parameters
        ----------
        R0 : float
            Equilibrium bubble radius  [m].
        G : float
            Shear modulus  [Pa].
        mu : float
            Dynamic viscosity  [Pa s]; default 0.89 mPa s.
        """
        return cls(R0=R0, G=G, mu=mu)

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return (4.0 / 3.0) * self.G * ((state.R / self.R0) ** 3 - 1.0)


class NeoHookeanMedium(MediumModel):
    """Neo-Hookean finite-strain viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R0/R)^3 - (R/R0)^3)

    Correctly captures finite-strain behaviour at large oscillation
    amplitudes where the Kelvin-Voigt linear approximation breaks down.

    Fields
    ------
    R0 : float
        Equilibrium bubble radius  [m].
    G : float
        Shear modulus  [Pa].
    mu : float
        Dynamic viscosity  [Pa s].
    """

    R0: float
    G: float
    mu: float

    @classmethod
    def water(
        cls,
        R0: float,
        G: float,
        mu: float = 0.00089,
    ) -> "NeoHookeanMedium":
        """Construct with water-like viscosity.

        Parameters
        ----------
        R0 : float
            Equilibrium bubble radius  [m].
        G : float
            Shear modulus  [Pa].
        mu : float
            Dynamic viscosity  [Pa s]; default 0.89 mPa s.
        """
        return cls(R0=R0, G=G, mu=mu)

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return (
            (4.0 / 3.0) * self.G * ((self.R0 / state.R) ** 3 - (state.R / self.R0) ** 3)
        )
