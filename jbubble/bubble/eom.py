from typing import Any, Callable

import jax
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


class LeightonTube(EquationOfMotion):
    """Leighton model for a bubble confined in a rigid-walled tube.

    Modifies the inertia terms of the standard Rayleigh-Plesset equation
    to account for the added mass effect of a rigid cylindrical tube::

        R Rddot [1 + (R/Gamma) beta]  +  3/2 Rdot^2 [1 + 4R/(3 Gamma) beta]
            = (1/rho) (p_L_damped - P_amb - p_ac)

    where ``beta = 2 alpha`` with
    ``alpha = (zeta/Gamma)(1 + 8 Gamma / (3 pi zeta)) - 1`` encodes
    the tube geometry (Gamma = tube radius, zeta = half-length).

    A simplified gas compressibility damping is included::

        p_gas_damped = p_gas(R) * (1 - 3 gamma Rdot / c_L)

    Fields
    ------
    gas : GasModel
    shell : ShellModel
    medium : MediumModel
    R0 : float
        Equilibrium bubble radius [m].
    P_amb : float
        Ambient pressure [Pa].
    rho_L : float
        Liquid density [kg/m^3].
    c_L : float
        Speed of sound in the liquid [m/s].
    tube_radius : float
        Inner radius of the confining tube [m].
    tube_length : float
        Length of the confining tube [m].
    """

    c_L: float
    tube_radius: float
    tube_length: float

    def p_L(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        """Boundary pressure with simplified gas compressibility damping.

        The gas pressure is multiplied by ``(1 - 3 gamma R_dot / c_L)``
        before entering the standard force balance.
        """
        damping = 1.0 - 3.0 * self.gas.gamma * R_dot / self.c_L
        return (
            self.gas(R) * damping
            - self.shell(R, R_dot)
            - self.medium(R, R_dot)
        )

    def __call__(
        self,
        t: Any,
        state: State,
        p_ac_fn: Callable,
    ) -> State:
        R, R_dot = state

        p_L_val = self.p_L(R, R_dot)
        p_ac = p_ac_fn(t)

        # Tube geometry factors
        Gamma = self.tube_radius
        zeta = self.tube_length / 2.0
        alpha = (
            (zeta / Gamma) * (1.0 + (8.0 * Gamma) / (3.0 * jnp.pi * zeta))
            - 1.0
        )
        beta = 2.0 * alpha

        rhs = (p_L_val - self.P_amb - p_ac) / self.rho_L

        # Modified inertia: R Rddot [1 + (R/Gamma) beta]
        #                 + 3/2 Rdot^2 [1 + 4R/(3 Gamma) beta] = rhs
        denom = R * (1.0 + (R / Gamma) * beta)
        inertia = 1.5 * R_dot**2 * (1.0 + (4.0 * R) / (3.0 * Gamma) * beta)

        R_ddot = (rhs - inertia) / denom
        return jnp.stack([R_dot, R_ddot])


class SphericalConfinement(EquationOfMotion):
    """Bubble confined inside a thin elastic spherical vessel.

    Couples the bubble wall dynamics to the vessel wall motion, producing
    a 4-DOF state vector ``[R, Rdot, a, a_dot]`` where *a* is the
    vessel wall radius.

    The coupled system is::

        A Rddot + B a_ddot = F   (force balance)
        C Rddot + D a_ddot = E   (vessel inertia)

    solved via Cramer's rule at each time step.

    A simplified gas compressibility damping is included::

        p_gas_damped = p_gas(R) * (1 - 3 gamma Rdot / c_L)

    Fields
    ------
    gas : GasModel
    shell : ShellModel
    medium : MediumModel
    R0 : float
        Equilibrium bubble radius [m].
    P_amb : float
        Ambient pressure [Pa].
    rho_L : float
        Liquid density [kg/m^3].
    c_L : float
        Speed of sound in the liquid [m/s].
    vessel_radius : float
        Equilibrium vessel wall radius [m].
    vessel_rho : float
        Vessel wall density [kg/m^3].
    vessel_E : float
        Vessel Young's modulus [Pa].
    vessel_d : float
        Vessel wall thickness [m].
    tissue_rho : float
        Surrounding tissue density [kg/m^3].
    tissue_d : float
        Surrounding tissue thickness [m].
    """

    c_L: float
    vessel_radius: float
    vessel_rho: float
    vessel_E: float
    vessel_d: float
    tissue_rho: float
    tissue_d: float

    def initial_state(self) -> State:
        """Initial state ``[R0, 0, vessel_radius, 0]``."""
        return jnp.array([self.R0, 0.0, self.vessel_radius, 0.0])

    def rescale_state(self, state: jax.Array, units) -> jax.Array:
        """Rescale 4-DOF state ``[R, Rdot, a, a_dot]`` to physical units."""
        scale_factors = jnp.array(
            [units.L_scale, units.vel_scale, units.L_scale, units.vel_scale]
        )
        return state * scale_factors

    def __call__(
        self,
        t: Any,
        state: State,
        p_ac_fn: Callable,
    ) -> State:
        R, R_dot, a, a_dot = state
        p_ac = p_ac_fn(t)

        # Gas pressure with simplified compressibility damping
        p_gas_damped = (
            self.gas(R) * (1.0 - 3.0 * self.gas.gamma * R_dot / self.c_L)
        )

        # Shell and medium contributions at the bubble wall
        p_shell = self.shell(R, R_dot)
        p_medium_visc = 4.0 * self.medium.mu * (R_dot / R + a_dot / a)

        # Vessel wall pressure (thin shell, nearly-incompressible nu=0.5)
        nu = 0.5
        a0 = self.vessel_radius
        P_wall = self.vessel_E * (a - a0) / ((1.0 - nu**2) * a**2)

        # 2x2 coupled system coefficients
        A = R**2
        B = -(a**2)
        C = self.rho_L * R**2 * (1.0 / R - 1.0 / a)
        D = self.vessel_rho * self.vessel_d + self.tissue_rho * self.tissue_d

        E = 2.0 * a * a_dot**2 - 2.0 * R * R_dot**2

        F = (
            p_gas_damped
            - 2.0 * R * R_dot * self.rho_L * (1.0 / R - 1.0 / a)
            - p_shell
            - p_medium_visc
            - P_wall
            - self.P_amb
            - p_ac
        )

        # Cramer's rule: Delta = A*D - B*C
        Delta = A * D - B * C
        Delta = jnp.where(jnp.abs(Delta) < 1e-14, 1e-14, Delta)

        R_ddot = (E * D - B * F) / Delta
        a_ddot = (A * F - C * E) / Delta

        return jnp.stack([R_dot, R_ddot, a_dot, a_ddot])
