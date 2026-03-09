from typing import Any, Callable

import jax.numpy as jnp

from .interfaces import EquationOfMotion, State


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
