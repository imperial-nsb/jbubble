"""
Concrete physics components implementing the modular interfaces.

Gas models
----------
- ``PolytropicGas``       p_gas(R) = P_gas0 (R0/R)^(3 gamma)
- ``VanDerWaalsGas``      Hard-core corrected polytropic gas

Surface tension models
----------------------
- ``ConstantSigma``       sigma(R) = const  (uncoated bubble)
- ``MarmottantSigma``     Piecewise Marmottant three-regime law
- ``GompertzSigma``       Smooth Gompertz approximation

Shell models
------------
- ``NoShell``             Laplace pressure only (no shell mechanics)
- ``LipidShell``          Thin lipid shell: Laplace + surface viscosity
- ``ThickShell``          Church (1995) thick viscoelastic shell

Medium models
-------------
- ``KelvinVoigtMedium``   Newtonian viscosity + linear Kelvin-Voigt elasticity
- ``NeoHookeanMedium``    Newtonian viscosity + finite-strain Neo-Hookean elasticity

Equations of motion
-------------------
- ``RayleighPlesset``         Incompressible-liquid EoM
- ``ModifiedRayleighPlesset`` RP with gas radiation damping (Marmottant 2005)
- ``KellerMiksis``            Compressible-liquid EoM with autodiff dp_L/dt
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from . import _defaults
from ._gompertz import gompertz_surface_tension
from .interfaces import (
    EquationOfMotion,
    MediumModel,
    ShellModel,
    State,
    SurfaceTensionModel,
)

# ---------------------------------------------------------------------------
# Surface tension models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shell models
# ---------------------------------------------------------------------------


class NoShell(ShellModel):
    """No shell coating — only Laplace pressure.

    p_shell = 2 sigma(R) / R

    Suitable for uncoated gas bubbles.  Pair with ``ConstantSigma``
    for a standard Rayleigh-Plesset setup, or with any other
    ``SurfaceTensionModel`` if desired.
    """

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return 2.0 * self.sigma(R) / R


class LipidShell(ShellModel):
    """Thin lipid shell with surface viscosity.

    p_shell = 2 sigma(R) / R  +  4 kappa_s Rdot / R^2

    This is the shell model used by Marmottant (2005) and most
    Gompertz-smoothed variants.  Pair with ``MarmottantSigma`` for
    the original piecewise law or ``GompertzSigma`` for smooth autodiff.

    Fields
    ------
    sigma : SurfaceTensionModel
        Surface tension law (inherited from ShellModel).
    kappa_s : float
        Shell surface-dilatational viscosity  [N s/m].
    """

    kappa_s: float = _defaults.KAPPA_S_LIPID

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return 2.0 * self.sigma(R) / R + 4.0 * self.kappa_s * R_dot / R**2


class ThickShell(ShellModel):
    """Church (1995) thick viscoelastic shell.

    In addition to Laplace pressure, this model includes thick-shell
    elastic and viscous contributions::

        p_elastic    = (4/3) G_s (d_s / R0) (1 - (R0/R)^3)
        p_shell_visc = 4 mu_s d_s Rdot / R^2

    Total shell pressure::

        p_shell = 2 sigma(R) / R  +  p_elastic  +  p_shell_visc

    Fields
    ------
    sigma : SurfaceTensionModel
        Surface tension law (inherited from ShellModel).
    R0 : float
        Equilibrium bubble radius  [m].
    d_s : float
        Shell thickness  [m].
    G_s : float
        Shell shear modulus  [Pa].
    mu_s : float
        Shell viscosity  [Pa s].
    """

    R0: float
    d_s: float = 4e-9
    G_s: float = 10e6
    mu_s: float = 0.5

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        p_laplace = 2.0 * self.sigma(R) / R
        p_elastic = (
            (4.0 / 3.0) * self.G_s * (self.d_s / self.R0) * (1.0 - (self.R0 / R) ** 3)
        )
        p_visc = 4.0 * self.mu_s * self.d_s * R_dot / R**2
        return p_laplace + p_elastic + p_visc


# ---------------------------------------------------------------------------
# Medium models
# ---------------------------------------------------------------------------


class KelvinVoigtMedium(MediumModel):
    """Kelvin-Voigt viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R / R0)^3 - 1)

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
        p_elastic = (4.0 / 3.0) * self.G * ((R / self.R0) ** 3 - 1.0)
        return p_viscous + p_elastic


class NeoHookeanMedium(MediumModel):
    """Neo-Hookean finite-strain viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R0/R)^3 - (R/R0)^3)

    Correctly captures finite-strain behaviour at large oscillation
    amplitudes where the Kelvin-Voigt linear approximation breaks down.
    When G = 0 (default), reduces to a Newtonian liquid.

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
        p_elastic = (4.0 / 3.0) * self.G * ((self.R0 / R) ** 3 - (R / self.R0) ** 3)
        return p_viscous + p_elastic


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------


class RayleighPlesset(EquationOfMotion):
    """Rayleigh-Plesset equation of motion (incompressible liquid).

    R Rddot + 3/2 Rdot^2 = (1/rho) (p_L - P_amb - p_ac)

    The simplest bubble dynamics EoM, assuming an incompressible
    surrounding liquid.  No additional fields beyond the base class.
    """

    def __call__(
        self,
        t: Any,
        state: State,
        p_ac_fn: Callable,
    ) -> State:
        R, R_dot = state
        p_L_val = self.p_L(R, R_dot)
        p_ac = p_ac_fn(t)
        R_ddot = ((p_L_val - self.P_amb - p_ac) / self.rho_L - 1.5 * R_dot**2) / R
        return jnp.stack([R_dot, R_ddot])


class ModifiedRayleighPlesset(EquationOfMotion):
    """Modified Rayleigh-Plesset with gas radiation damping.

    Adds a first-order compressibility correction to the gas pressure
    term only, as used by Marmottant et al. (2005)::

        R Rddot + 3/2 Rdot^2
            = (1/rho) (p_L + (R/c) dp_gas/dt - P_amb - p_ac)

    where dp_gas/dt = (dp_gas/dR) Rdot is computed via autodiff.  This
    sits between the plain Rayleigh-Plesset (no compressibility) and
    the full Keller-Miksis (first-order compressibility on all of p_L).

    Fields
    ------
    gas : GasModel
    shell : ShellModel
    medium : MediumModel
    R0 : float
        Equilibrium bubble radius  [m].
    P_amb : float
        Ambient pressure  [Pa].
    rho_L : float
        Liquid density  [kg/m^3].
    c_L : float
        Speed of sound in the liquid  [m/s].
    """

    c_L: float

    def __call__(
        self,
        t: Any,
        state: State,
        p_ac_fn: Callable,
    ) -> State:
        R, R_dot = state

        p_L_val = self.p_L(R, R_dot)
        p_ac = p_ac_fn(t)

        # Gas radiation damping: dp_gas/dt = (dp_gas/dR) * Rdot
        dp_gas_dR = jax.grad(self.gas)(R)
        dp_gas_dt = dp_gas_dR * R_dot

        forces = p_L_val + (R / self.c_L) * dp_gas_dt - self.P_amb - p_ac
        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R
        return jnp.stack([R_dot, R_ddot])


class KellerMiksis(EquationOfMotion):
    """Keller-Miksis equation of motion (first-order compressibility).

    Accounts for liquid compressibility up to first order in the Mach
    number M = Rdot / c_L::

        (1-M) R Rddot + 3/2 (1 - M/3) Rdot^2
            = (1/rho) (1+M) (p_L - P_amb - p_ac)
              + R / (rho c) (dp_L/dt - dp_ac/dt)

    The time derivative dp_L/dt is computed **automatically** via JAX
    autodiff (chain rule through ``p_L``), so this EoM works with any
    combination of gas, shell, and medium models without hand-coded
    derivatives.

    Fields
    ------
    gas : GasModel
    shell : ShellModel
    medium : MediumModel
    R0 : float
        Equilibrium bubble radius  [m].
    P_amb : float
        Ambient pressure  [Pa].
    rho_L : float
        Liquid density  [kg/m^3].
    c_L : float
        Speed of sound in the liquid  [m/s].
    """

    c_L: float

    def __call__(
        self,
        t: Any,
        state: State,
        p_ac_fn: Callable,
    ) -> State:
        R, R_dot = state
        M = R_dot / self.c_L  # Mach number

        # -- boundary pressure and its partial derivatives (autodiff) ------
        p_L_val = self.p_L(R, R_dot)
        dp_L_dR = jax.grad(self.p_L, argnums=0)(R, R_dot)
        dp_L_dRdot = jax.grad(self.p_L, argnums=1)(R, R_dot)

        # -- driving pressure and its time derivative ----------------------
        p_ac = p_ac_fn(t)
        dp_ac_dt = jax.grad(p_ac_fn)(t)

        # -- Keller-Miksis: collect Rddot on the LHS ----------------------
        #
        # dp_L/dt = (dp_L/dR) Rdot  +  (dp_L/dRdot) Rddot
        #                ^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^
        #                numer term       absorbed into denom
        #
        # denom * Rddot = numer

        denom = (1.0 - M) * R - (R / (self.rho_L * self.c_L)) * dp_L_dRdot

        numer = (
            (1.0 / self.rho_L) * (1.0 + M) * (p_L_val - self.P_amb - p_ac)
            + (R / (self.rho_L * self.c_L)) * (dp_L_dR * R_dot - dp_ac_dt)
            - 1.5 * (1.0 - M / 3.0) * R_dot**2
        )

        R_ddot = numer / denom
        return jnp.stack([R_dot, R_ddot])
