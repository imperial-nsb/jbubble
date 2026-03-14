"""State-dependent material property modules.

The ``Property`` pattern replaces static ``float`` parameters with
explicit Equinox modules whose ``__call__`` maps a ``BubbleState`` to a
scalar value.  This allows users to inject arbitrary non-linear rheology
or even internal neural networks, seamlessly compatible with JAX autodiff.

Surface tension models previously in ``surface.py`` are reimplemented
here as ``Property`` subclasses.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .state import BubbleState


class Property(eqx.Module):
    """A property that returns a constant value regardless of state.

    Fields
    ------
    val : float
        The constant value.
    """

    val: float = eqx.field(default_factory=lambda: 0.0, kw_only=True)

    def __call__(self, state: BubbleState) -> jax.Array:
        return self.val + state.R * 0.0


def as_property(val: "float | Property") -> "Property":
    """Coerce a plain float to a ``Property``, or pass through a ``Property``.

    Parameters
    ----------
    val : float or Property
        A plain scalar or an existing ``Property`` instance.

    Returns
    -------
    Property
    """
    if isinstance(val, Property):
        return val
    return Property(val=float(val))


class GompertzSurfaceTension(Property):
    """Smooth Gompertz surface tension law.

    A differentiable Gompertz function approximates the piecewise
    Marmottant surface tension, enabling robust automatic
    differentiation::

        sigma(R) = a exp(-b exp(c (1 - R / R_buckle)))

    where R_buckle = R_buckle_ratio * state.R0.

    The Gompertz parameters b and c are derived from chi and sigma_break
    such that sigma(R0) matches the elastic regime and sigma -> sigma_break
    as R -> infinity.  R0 is read from the state so this model stays
    consistent when R0 evolves (e.g. rectified diffusion).

    Fields
    ------
    R_buckle_ratio : float
        Buckling radius as a fraction of R0  (dimensionless).
    chi : float or Property
        Shell elasticity  [N/m].
    sigma_break : float or Property
        Asymptotic (ruptured) surface tension  [N/m].
    """

    R_buckle_ratio: float
    chi: float
    sigma_break: float

    def __call__(self, state: BubbleState) -> jax.Array:
        R = state.R
        R0 = state.R0
        R_buckle = self.R_buckle_ratio * R0
        chi = self.chi
        a = self.sigma_break
        c = (2.0 * chi / a) * jnp.sqrt(1.0 + a / (2.0 * chi))
        sigma_R0 = chi * ((R0 / R_buckle) ** 2 - 1.0)
        b = -jnp.log(sigma_R0 / a) / jnp.exp(c * (1.0 - R0 / R_buckle))
        return a * jnp.exp(-b * jnp.exp(c * (1.0 - R / R_buckle)))


class MarmottantSurfaceTension(Property):
    """Piecewise Marmottant surface tension law.

    Three regimes based on the ratio R / R_buckle::

        R <= R_buckle                :  sigma = 0               (buckled)
        R_buckle < R < R_rupture     :  sigma = chi ((R/R_b)^2 - 1)  (elastic)
        R >= R_rupture               :  sigma = sigma_rupture   (ruptured)

    where R_buckle = R_buckle_ratio * state.R0 and R_rupture is derived
    from continuity of sigma at the elastic-to-ruptured transition.

    Note: sigma(R) has discontinuous first derivatives at the regime
    boundaries.  For applications requiring smooth gradients (e.g.
    gradient-based optimisation), use ``GompertzSurfaceTension`` instead.

    Fields
    ------
    R_buckle_ratio : float
        Buckling radius as a fraction of R0  (dimensionless).
    chi : float or Property
        Shell elasticity  [N/m].
    sigma_rupture : float or Property
        Surface tension (post-rupture value)  [N/m].
    """

    R_buckle_ratio: float
    chi: float
    sigma_rupture: float

    def __call__(self, state: BubbleState) -> jax.Array:
        R = state.R
        R_buckle = self.R_buckle_ratio * state.R0
        chi = self.chi
        sigma_rupture = self.sigma_rupture
        R_rupture = R_buckle * jnp.sqrt(sigma_rupture / chi + 1.0)
        sigma_elastic = chi * ((R / R_buckle) ** 2 - 1.0)
        in_elastic = (R_buckle < R) & (R_rupture > R)
        in_ruptured = R_rupture <= R
        return jnp.where(
            in_ruptured,
            sigma_rupture,
            jnp.where(
                in_elastic,
                sigma_elastic,
                0.0,
            ),
        )
