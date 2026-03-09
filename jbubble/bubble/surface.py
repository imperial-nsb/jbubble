import jax
import jax.numpy as jnp

from . import _defaults
from .interfaces import SurfaceTensionModel


def gompertz_surface_tension(
    R,
    *,
    R0: float,
    R_buckle: float,
    chi: float,
    sigma_break: float,
    sigma_R0: float,
):
    """
    Evaluate the Gompertz surface tension at radius *R*.

    The Gompertz function provides a smooth, differentiable approximation to the
    piecewise Marmottant surface tension law, enabling automatic differentiation.

    σ(R) = a · exp(−b · exp(c · (1 − R/R_buckle)))

    where a, b, c are derived from the shell parameters so that:
    - σ → 0        as R → −∞  (buckled limit)
    - σ = σ_R0     at R = R0  (equilibrium)
    - σ → σ_break  as R → +∞  (ruptured limit)

    Parameters
    ----------
    R : scalar or array
        Bubble radius.
    R0 : float
        Equilibrium radius.
    R_buckle : float
        Buckling radius (transition from buckled to elastic regime).
    chi : float
        Shell elasticity modulus [N/m].
    sigma_break : float
        Surface tension at shell rupture [N/m].
    sigma_R0 : float
        Surface tension at equilibrium (pre-computed as χ(R0²/R_buckle² − 1)).

    Returns
    -------
    Same type as *R*.
    """
    a = sigma_break
    c = (2.0 * chi / a) * jnp.sqrt(1.0 + a / (2.0 * chi))
    b = -jnp.log(sigma_R0 / a) / jnp.exp(c * (1.0 - R0 / R_buckle))
    return a * jnp.exp(-b * jnp.exp(c * (1.0 - R / R_buckle)))


class ConstantSigma(SurfaceTensionModel):
    """Constant surface tension (uncoated bubble).

    sigma(R) = sigma_L  (independent of R)

    Fields
    ------
    sigma_L : float
        Liquid surface tension  [N/m].
    """

    sigma_L: float = _defaults.SIGMA_WATER

    def __call__(self, R: jax.Array) -> jax.Array:
        return self.sigma_L + R * 0.0


class MarmottantSigma(SurfaceTensionModel):
    """Piecewise Marmottant surface tension law.

    Three regimes based on the ratio R / R_buckle::

        R <= R_buckle                :  sigma = 0               (buckled)
        R_buckle < R < R_rupture     :  sigma = chi ((R/R_b)^2 - 1)  (elastic)
        R >= R_rupture               :  sigma = sigma_water     (ruptured)

    Note: sigma(R) has discontinuous first derivatives at the regime
    boundaries.  For applications requiring smooth gradients (e.g.
    gradient-based optimisation), use ``GompertzSigma`` instead.

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
    chi: float = _defaults.CHI_LIPID
    sigma_water: float = _defaults.SIGMA_WATER

    @classmethod
    def from_R0(
        cls,
        *,
        R0: float,
        R_buckle_ratio: float = _defaults.R_BUCKLE_RATIO,
        **kwargs,
    ) -> "MarmottantSigma":
        """Construct from equilibrium radius and buckling ratio."""
        return cls(R_buckle=R0 * R_buckle_ratio, **kwargs)

    @property
    def R_rupture(self) -> float:
        """Rupture radius, derived from continuity of sigma at rupture."""
        return self.R_buckle * jnp.sqrt(self.sigma_water / self.chi + 1.0)

    def sigma_R0(self, R0: float) -> float:
        """Surface tension at a given R0."""
        return self.chi * ((R0 / self.R_buckle) ** 2 - 1.0)

    def __call__(self, R: jax.Array) -> jax.Array:
        sigma_elastic = self.chi * ((R / self.R_buckle) ** 2 - 1.0)
        return jnp.where(
            R <= self.R_buckle,
            0.0,
            jnp.where(R >= self.R_rupture, self.sigma_water, sigma_elastic),
        )


class GompertzSigma(SurfaceTensionModel):
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
    chi: float = _defaults.CHI_LIPID
    sigma_break: float = _defaults.SIGMA_WATER

    @classmethod
    def from_R0(
        cls,
        *,
        R0: float,
        R_buckle_ratio: float = _defaults.R_BUCKLE_RATIO,
        **kwargs,
    ) -> "GompertzSigma":
        """Construct from equilibrium radius and buckling ratio."""
        return cls(R0=R0, R_buckle=R0 * R_buckle_ratio, **kwargs)

    @property
    def sigma_R0(self) -> float:
        """Surface tension at R0, from the elastic regime formula."""
        return self.chi * ((self.R0 / self.R_buckle) ** 2 - 1.0)

    def __call__(self, R: jax.Array) -> jax.Array:
        return gompertz_surface_tension(
            R,
            R0=self.R0,
            R_buckle=self.R_buckle,
            chi=self.chi,
            sigma_break=self.sigma_break,
            sigma_R0=self.sigma_R0,
        )
