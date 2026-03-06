"""
Concrete physics components implementing the modular interfaces.

Gas models
----------
- ``PolytropicGas``       p_gas(R) = P_gas0 (R0/R)^(3 gamma)

Shell models
------------
- ``MarmottantShell``     Piecewise Marmottant surface tension + shell viscosity

Medium models
-------------
- ``KelvinVoigtMedium``   Newtonian viscosity + linear Kelvin-Voigt elasticity
"""

import jax
import jax.numpy as jnp

from . import _defaults
from .interfaces import GasModel, MediumModel, ShellModel

# ---------------------------------------------------------------------------
# Gas models
# ---------------------------------------------------------------------------


class PolytropicGas(GasModel):
    """Polytropic gas law.

    p_gas(R) = P_gas0 (R0 / R)^(3 gamma)

    Fields
    ------
    P_gas0 : float
        Equilibrium gas pressure  [Pa].
    R0 : float
        Equilibrium bubble radius  [m].
    gamma : float
        Polytropic exponent  (1.0 = isothermal, 1.4 = adiabatic air).
    """

    P_gas0: float
    R0: float
    gamma: float

    @classmethod
    def from_equilibrium(
        cls,
        *,
        R0: float,
        gamma: float,
        P_amb: float,
        sigma_R0: float,
    ) -> "PolytropicGas":
        """Construct from physical equilibrium conditions.

        Computes  P_gas0 = P_amb + 2 sigma(R0) / R0.

        Parameters
        ----------
        R0 : float
            Equilibrium radius  [m].
        gamma : float
            Polytropic exponent.
        P_amb : float
            Ambient pressure  [Pa].
        sigma_R0 : float
            Surface tension evaluated at R0  [N/m].
        """
        P_gas0 = P_amb + 2.0 * sigma_R0 / R0
        return cls(P_gas0=P_gas0, R0=R0, gamma=gamma)

    def __call__(self, R: jax.Array) -> jax.Array:
        return self.P_gas0 * (self.R0 / R) ** (3.0 * self.gamma)


# ---------------------------------------------------------------------------
# Shell models
# ---------------------------------------------------------------------------


class MarmottantShell(ShellModel):
    """Marmottant piecewise lipid shell model.

    Surface tension follows three regimes::

        R <= R_buckle                :  sigma = 0
        R_buckle < R < R_rupture     :  sigma = chi ((R / R_buckle)^2 - 1)
        R >= R_rupture               :  sigma = sigma_water

    Total shell pressure::

        p_shell = 2 sigma(R) / R  +  4 kappa_s Rdot / R^2

    Note: sigma(R) has discontinuous first derivatives at the regime
    boundaries.  For applications requiring smooth gradients (e.g.
    gradient-based optimisation), use a Gompertz-smoothed shell instead.

    Fields
    ------
    R_buckle : float
        Buckling radius  [m].
    chi : float
        Shell elasticity  [N/m].
    kappa_s : float
        Shell surface-dilatational viscosity  [N s/m].
    sigma_water : float
        Water surface tension (post-rupture value)  [N/m].
    """

    R_buckle: float
    chi: float = _defaults.CHI_LIPID
    kappa_s: float = _defaults.KAPPA_S_LIPID
    sigma_water: float = _defaults.SIGMA_WATER

    @classmethod
    def from_R0(
        cls,
        *,
        R0: float,
        R_buckle_ratio: float = _defaults.R_BUCKLE_RATIO,
        **kwargs,
    ) -> "MarmottantShell":
        """Construct from equilibrium radius and buckling ratio.

        Parameters
        ----------
        R0 : float
            Equilibrium bubble radius  [m].
        R_buckle_ratio : float
            R_buckle / R0 ratio  (default 0.99).
        **kwargs
            Forwarded to the constructor (chi, kappa_s, sigma_water).
        """
        return cls(R_buckle=R0 * R_buckle_ratio, **kwargs)

    @property
    def R_rupture(self) -> float:
        """Rupture radius, derived from continuity of sigma at rupture."""
        return self.R_buckle * jnp.sqrt(self.sigma_water / self.chi + 1.0)

    def surface_tension(self, R: jax.Array) -> jax.Array:
        sigma_elastic = self.chi * ((R / self.R_buckle) ** 2 - 1.0)
        return jnp.where(
            R <= self.R_buckle,
            0.0,
            jnp.where(R >= self.R_rupture, self.sigma_water, sigma_elastic),
        )

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        sigma = self.surface_tension(R)
        return 2.0 * sigma / R + 4.0 * self.kappa_s * R_dot / R**2


# ---------------------------------------------------------------------------
# Medium models
# ---------------------------------------------------------------------------


class KelvinVoigtMedium(MediumModel):
    """Kelvin-Voigt viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) (1 - (R0 / R)^3)

    When G = 0 (default), reduces to a Newtonian liquid:  4 mu Rdot / R.

    Fields
    ------
    R0 : float
        Equilibrium bubble radius  [m].
    mu : float
        Dynamic viscosity  [Pa s].
    G : float
        Shear modulus  [Pa].  Zero for a Newtonian liquid.
    """

    R0: float
    mu: float = _defaults.MU_WATER
    G: float = 0.0

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        p_viscous = 4.0 * self.mu * R_dot / R
        p_elastic = (4.0 / 3.0) * self.G * (1.0 - (self.R0 / R) ** 3)
        return p_viscous + p_elastic
