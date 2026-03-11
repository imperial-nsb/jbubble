"""State-dependent material property modules.

The ``Property`` pattern replaces static ``float`` parameters with
explicit Equinox modules whose ``__call__`` maps a ``BubbleState`` to a
scalar value.  This allows users to inject arbitrary non-linear rheology
or even internal neural networks, seamlessly compatible with JAX autodiff.

Surface tension models previously in ``surface.py`` are reimplemented
here as ``Property`` subclasses.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .state import BubbleState


class Property(eqx.Module, abc.ABC):
    """Base class for state-dependent material properties.

    A ``Property`` maps the current bubble state to a scalar physical
    quantity (surface tension, shell viscosity, etc.).
    """

    @abc.abstractmethod
    def __call__(self, state: BubbleState) -> jax.Array:
        """Evaluate the property at the given state.

        Parameters
        ----------
        state : BubbleState
            Current bubble state.

        Returns
        -------
        scalar
            Property value.
        """
        ...


class ConstantProperty(Property):
    """A property that returns a constant value regardless of state.

    Fields
    ------
    val : float
        The constant value.
    """

    val: float
    _scale: str = eqx.field(static=True, default="unit_scale")

    def __call__(self, state: BubbleState) -> jax.Array:
        return self.val + state.R * 0.0


class GompertzSurfaceTension(Property):
    """Smooth Gompertz surface tension law.

    A differentiable Gompertz function approximates the piecewise
    Marmottant surface tension, enabling robust automatic
    differentiation::

        sigma(R) = a exp(-b exp(c (1 - R / R_buckle)))

    Fields
    ------
    R0 : float
        Equilibrium bubble radius  [m].
    R_buckle : float
        Buckling radius  [m].
    chi : float
        Shell elasticity  [N/m].
    sigma_break : float
        Asymptotic (ruptured) surface tension  [N/m].
    """

    R0: float
    R_buckle: float
    chi: float
    sigma_break: float

    @classmethod
    def from_R0(
        cls,
        *,
        R0: float,
        R_buckle_ratio: float = 0.99,
        chi: float = 0.38,
        sigma_break: float = 72e-3,
    ) -> "GompertzSurfaceTension":
        """Construct from equilibrium radius and buckling ratio."""
        return cls(R0=R0, R_buckle=R0 * R_buckle_ratio, chi=chi, sigma_break=sigma_break)

    @property
    def sigma_R0(self) -> float:
        """Surface tension at R0, from the elastic regime formula."""
        return self.chi * ((self.R0 / self.R_buckle) ** 2 - 1.0)

    def __call__(self, state: BubbleState) -> jax.Array:
        R = state.R
        a = self.sigma_break
        c = (2.0 * self.chi / a) * jnp.sqrt(1.0 + a / (2.0 * self.chi))
        b = -jnp.log(self.sigma_R0 / a) / jnp.exp(
            c * (1.0 - self.R0 / self.R_buckle)
        )
        return a * jnp.exp(-b * jnp.exp(c * (1.0 - R / self.R_buckle)))


class MarmottantSurfaceTension(Property):
    """Piecewise Marmottant surface tension law.

    Three regimes based on the ratio R / R_buckle::

        R <= R_buckle                :  sigma = 0               (buckled)
        R_buckle < R < R_rupture     :  sigma = chi ((R/R_b)^2 - 1)  (elastic)
        R >= R_rupture               :  sigma = sigma_water     (ruptured)

    Note: sigma(R) has discontinuous first derivatives at the regime
    boundaries.  For applications requiring smooth gradients (e.g.
    gradient-based optimisation), use ``GompertzSurfaceTension`` instead.

    Fields
    ------
    R_buckle : float
        Buckling radius  [m].
    chi : float
        Shell elasticity  [N/m].
    sigma_water : float
        Water surface tension (post-rupture value)  [N/m].
    """

    R_buckle: float
    chi: float
    sigma_water: float

    @classmethod
    def from_R0(
        cls,
        *,
        R0: float,
        R_buckle_ratio: float = 0.99,
        chi: float = 0.38,
        sigma_water: float = 72e-3,
    ) -> "MarmottantSurfaceTension":
        """Construct from equilibrium radius and buckling ratio."""
        return cls(R_buckle=R0 * R_buckle_ratio, chi=chi, sigma_water=sigma_water)

    @property
    def R_rupture(self) -> float:
        """Rupture radius, derived from continuity of sigma at rupture."""
        return self.R_buckle * jnp.sqrt(self.sigma_water / self.chi + 1.0)

    def sigma_R0(self, R0: float) -> float:
        """Surface tension at a given R0."""
        return self.chi * ((R0 / self.R_buckle) ** 2 - 1.0)

    def __call__(self, state: BubbleState) -> jax.Array:
        R = state.R
        sigma_elastic = self.chi * ((R / self.R_buckle) ** 2 - 1.0)
        return jnp.where(
            R <= self.R_buckle,
            0.0,
            jnp.where(R >= self.R_rupture, self.sigma_water, sigma_elastic),
        )
