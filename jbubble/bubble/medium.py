"""Surrounding medium (fluid / tissue) models.

Computes the total inward viscous and elastic stresses exerted by the
surrounding medium on the bubble wall.
"""

import abc
import dataclasses

import equinox as eqx
import jax

from .properties import Property, as_property
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
    ``ConstantProperty`` in ``__post_init__``.

    Subclasses can declare additional ``Property`` fields by annotating
    them with ``eqx.field(metadata={"property_scale": "<scale_name>"})``
    — the base ``__post_init__`` will convert them automatically.
    No subclass ``__post_init__`` is needed.

    Fields
    ------
    mu : float or Property
        Dynamic viscosity  [Pa s].
    """

    mu: Property = eqx.field(metadata={"property_scale": "mu_scale"})

    def __post_init__(self):
        for f in dataclasses.fields(self):
            scale = f.metadata.get("property_scale")
            if scale is not None:
                object.__setattr__(
                    self, f.name, as_property(getattr(self, f.name), scale)
                )

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

    R0: float
    G: Property = eqx.field(metadata={"property_scale": "P_scale"})

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu(state) * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return (4.0 / 3.0) * self.G(state) * ((state.R / self.R0) ** 3 - 1.0)


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

    R0: float
    G: Property  # pass a plain float and it is auto-converted in __post_init__

    def __post_init__(self):
        super().__post_init__()  # converts mu
        object.__setattr__(self, "G", as_property(self.G, "P_scale"))

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu(state) * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return (
            (4.0 / 3.0)
            * self.G(state)
            * ((self.R0 / state.R) ** 3 - (state.R / self.R0) ** 3)
        )
