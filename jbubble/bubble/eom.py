"""Macroscopic equations of motion for bubble dynamics.

Each concrete ``EquationOfMotion`` assembles a ``GasModel``,
``ShellModel``, and ``MediumModel`` into a complete ODE right-hand side
that returns a ``BubbleState`` (time derivative).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .gas import GasModel
from .medium import MediumModel
from .shell import ShellModel
from .state import BubbleState

StateType = TypeVar("StateType", bound=BubbleState)


class EquationOfMotion(eqx.Module, abc.ABC, Generic[StateType]):
    """Macroscopic equation of motion for bubble dynamics.

    Assembles a ``GasModel``, ``ShellModel``, and ``MediumModel`` into a
    complete ODE right-hand side.  Concrete subclasses encode different
    EoM formulations (Rayleigh-Plesset, Keller-Miksis, ...) which differ
    in how they relate p_L to the radial acceleration Rddot.

    The type parameter ``StateType`` binds each EoM to the state it operates
    on.  The built-in EoMs use ``BubbleState``; an EoM that carries extra
    degrees of freedom binds a ``BubbleState`` subclass, so passing the
    wrong state type is caught by the type checker rather than failing at
    runtime deep inside a JAX trace.

    The liquid-side boundary pressure is computed by the concrete helper
    ``p_L`` as::

        p_L = gas(state) - shell(state) - medium(state)

    Because ``p_L`` is a regular method on an Equinox module, JAX can
    differentiate through it automatically.  EoMs that require dp_L/dt
    (e.g. Keller-Miksis) can use ``jax.grad(self.p_L)`` instead of
    hand-coding analytical derivatives for every component combination.

    The driving acoustic pressure is received as a callable ``p_ac_fn``
    so that EoMs needing dp_ac/dt can compute it via ``jax.grad``.

    ``R0`` and ``P_gas0`` (equilibrium configuration) are seeded into the
    state by ``initial_state()`` and carried as frozen constants by the
    ODE (zero time derivatives in the standard case).

    Fields
    ------
    gas : GasModel
        Internal gas pressure model.
    shell : ShellModel
        Shell / coating model.
    medium : MediumModel
        Surrounding medium model.
    R0 : float or jax.Array
        Equilibrium bubble radius  [m].
    P_amb : float or jax.Array
        Ambient (far-field) pressure  [Pa].
    rho_L : float or jax.Array
        Liquid density  [kg/m^3].
    """

    gas: GasModel
    shell: ShellModel
    medium: MediumModel

    R0: ArrayLike
    P_amb: ArrayLike
    rho_L: ArrayLike

    def p_L(self, state: BubbleState) -> jax.Array:
        """Liquid-side boundary pressure.

        p_L = p_gas(state) - p_shell(state) - p_medium(state)
        """
        return self.gas(state) - self.shell(state) - self.medium(state)

    def initial_state(self) -> BubbleState:
        """Default initial state: equilibrium radius, zero velocity.

        Seeds ``R0`` and ``P_gas0`` (Laplace equilibrium) into the state.
        Override for coupled systems with a larger state vector.
        """
        R0 = jnp.asarray(self.R0)
        P_gas0 = (
            jnp.asarray(self.P_amb)
            + 2.0 * self.shell.sigma(BubbleState(R=R0, R0=R0)) / R0
        )
        return BubbleState(R=R0, R0=R0, P_gas0=P_gas0)

    @abc.abstractmethod
    def __call__(
        self,
        t: Any,
        state: StateType,
        p_ac_fn: Callable,
    ) -> StateType:
        """Compute the ODE right-hand side  d(state)/dt.

        Parameters
        ----------
        t : scalar
            Current time.
        state : StateType
            Current bubble state (type matches the EoM's state parameter).
        p_ac_fn : callable  (t -> scalar)
            Driving acoustic pressure as a function of time.

        Returns
        -------
        StateType
            Time derivative of the state.  ``R0`` and ``P_gas0``
            derivatives are zero in the standard case.
        """
        ...


class RayleighPlesset(EquationOfMotion[BubbleState]):
    """Rayleigh-Plesset equation of motion (incompressible liquid).

    R Rddot + 3/2 Rdot^2 = (1/rho) (p_L - P_amb - p_ac)

    The simplest bubble dynamics EoM, assuming an incompressible
    surrounding liquid.  No additional fields beyond the base class.
    """

    def __call__(
        self,
        t: Any,
        state: BubbleState,
        p_ac_fn: Callable,
    ) -> BubbleState:
        R, R_dot = state.R, state.R_dot
        p_L_val = self.p_L(state)
        p_ac = p_ac_fn(t)
        R_ddot = ((p_L_val - self.P_amb - p_ac) / self.rho_L - 1.5 * R_dot**2) / R
        return BubbleState(R=R_dot, R_dot=R_ddot)


class ModifiedRayleighPlesset(EquationOfMotion[BubbleState]):
    """Modified Rayleigh-Plesset with gas radiation damping.

    Adds a first-order compressibility correction to the gas pressure
    term only, as used by Marmottant et al. (2005)::

        R Rddot + 3/2 Rdot^2
            = (1/rho) (p_L + (R/c) dp_gas/dt - P_amb - p_ac)

    where dp_gas/dt = (dp_gas/dR) Rdot is computed via autodiff.

    Fields
    ------
    c_L : float or jax.Array
        Speed of sound in the liquid  [m/s].
    """

    c_L: ArrayLike

    def __call__(
        self,
        t: Any,
        state: BubbleState,
        p_ac_fn: Callable,
    ) -> BubbleState:
        R, R_dot = state.R, state.R_dot

        p_L_val = self.p_L(state)
        p_ac = p_ac_fn(t)

        # Gas radiation damping: dp_gas/dt = (dp_gas/dR) * Rdot
        gas_tangent = jax.grad(self.gas)(state)
        dp_gas_dR = gas_tangent.R
        dp_gas_dt = dp_gas_dR * R_dot

        forces = p_L_val + (R / self.c_L) * dp_gas_dt - self.P_amb - p_ac
        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R
        return BubbleState(R=R_dot, R_dot=R_ddot)


class KellerMiksis(EquationOfMotion[BubbleState]):
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
    c_L : float or jax.Array
        Speed of sound in the liquid  [m/s].
    """

    c_L: ArrayLike

    def __call__(
        self,
        t: Any,
        state: BubbleState,
        p_ac_fn: Callable,
    ) -> BubbleState:
        R, R_dot = state.R, state.R_dot
        M = R_dot / self.c_L  # Mach number

        # -- boundary pressure and its partial derivatives (autodiff) ------
        p_L_val = self.p_L(state)
        tangent = jax.grad(self.p_L)(state)
        dp_L_dR = tangent.R
        dp_L_dRdot = tangent.R_dot

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
        return BubbleState(R=R_dot, R_dot=R_ddot)


class Gilmore(EquationOfMotion[BubbleState]):
    """Gilmore equation of motion (Kirkwood-Bethe hypothesis).

    Improves on Keller-Miksis by treating liquid compressibility through
    the Tait equation of state rather than a linear approximation.  The
    enthalpy H and local speed of sound C at the bubble wall are exact
    functions of the wall pressure, giving accurate results at high Mach
    numbers::

        (1 - Ṙ/C) R R̈  +  3/2 (1 - Ṙ/(3C)) Ṙ²
            = (1 + Ṙ/C) H  +  (R/C)(1 - Ṙ/C) Ḣ

    where the enthalpy difference between bubble wall and far field is::

        H = n/(n-1) · K · [(p_L+B)^((n-1)/n) − (p∞+B)^((n-1)/n)]
        K = (P_amb+B)^(1/n) / ρ_L

    and the local sound speed satisfies::

        C² = n·K·(p∞+B)^((n-1)/n)  +  (n-1)·H

    with p∞ = P_amb + p_ac.

    Ḣ is expanded analytically via the chain rule: ∂H/∂p_L and ∂H/∂p∞
    come from the Tait formula; dp_L/dt is obtained via
    ``jax.grad(self.p_L)`` and dp_ac/dt via ``jax.grad(p_ac_fn)``.
    The resulting coupling of R̈ through dp_L/dṘ is absorbed into the
    denominator, following the same algebraic pattern as ``KellerMiksis``.

    Default Tait parameters correspond to water (Gilmore 1952):
    n = 7, B = 304.9 MPa.

    References
    ----------
    Gilmore, F. R. (1952). *The growth or collapse of a spherical bubble
    in a viscous compressible liquid.* Hydrodynamics Laboratory Report
    26-4, California Institute of Technology.

    Fields
    ------
    n_tait : float or jax.Array
        Tait exponent (dimensionless).  Default 7.0.
    B_tait : float or jax.Array
        Tait pressure constant  [Pa].  Default 304.9e6.
    """

    n_tait: ArrayLike = 7.0
    B_tait: ArrayLike = 304.9e6

    def _tait_K(self) -> jax.Array:
        """Tait EOS prefactor: (P_amb + B)^(1/n) / rho_L."""
        return jnp.asarray(
            (self.P_amb + self.B_tait) ** (1.0 / self.n_tait) / self.rho_L
        )

    def _H_and_C(self, p_L: jax.Array, p_inf: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Enthalpy H [m²/s²] and bubble-wall sound speed C [m/s]."""
        n, B = jnp.asarray(self.n_tait), jnp.asarray(self.B_tait)
        K = self._tait_K()
        exp = (n - 1.0) / n
        H = n / (n - 1.0) * K * ((p_L + B) ** exp - (p_inf + B) ** exp)
        c_inf_sq = n * K * (p_inf + B) ** exp
        C = jnp.sqrt(jnp.maximum(c_inf_sq + (n - 1.0) * H, 1.0))
        return H, C

    def __call__(
        self,
        t: Any,
        state: BubbleState,
        p_ac_fn: Callable,
    ) -> BubbleState:
        R, R_dot = state.R, state.R_dot

        p_L_val = self.p_L(state)
        p_ac = p_ac_fn(t)
        p_inf = self.P_amb + p_ac

        # Enthalpy and local sound speed at the bubble wall
        H, C = self._H_and_C(p_L_val, p_inf)
        M = R_dot / C

        # ∂H/∂p_L and ∂H/∂p∞ — analytical from the Tait formula
        n, B = self.n_tait, self.B_tait
        K = self._tait_K()
        h_pL = K * (p_L_val + B) ** (-1.0 / n)
        h_pinf = -K * (p_inf + B) ** (-1.0 / n)

        # ∂p_L/∂R and ∂p_L/∂Ṙ via autodiff
        tangent = jax.grad(self.p_L)(state)
        dp_L_dR = tangent.R
        dp_L_dRdot = tangent.R_dot

        # dp_ac/dt via autodiff
        dp_ac_dt = jax.grad(p_ac_fn)(t)

        # Ḣ = h_pL·(dp_L/dR·Ṙ + dp_L/dṘ·R̈) + h_pinf·dp_ac/dt
        # Split into the part independent of R̈ and the coefficient of R̈
        Hdot_numer = h_pL * dp_L_dR * R_dot + h_pinf * dp_ac_dt
        Hdot_Rddot_coeff = h_pL * dp_L_dRdot

        # Collect R̈ on the left-hand side: denom·R̈ = numer
        #
        # LHS: (1-M) R R̈
        # RHS R̈ term: (R/C)(1-M) · Hdot_Rddot_coeff · R̈
        # => denom = (1-M) R - (R/C)(1-M) · Hdot_Rddot_coeff
        factor = (R / C) * (1.0 - M)  # (R/C)(1 - Ṙ/C)

        numer = (1.0 + M) * H + factor * Hdot_numer - 1.5 * (1.0 - M / 3.0) * R_dot**2
        denom = (1.0 - M) * R - factor * Hdot_Rddot_coeff

        R_ddot = numer / denom
        return BubbleState(R=R_dot, R_dot=R_ddot)
