"""Macroscopic equations of motion for bubble dynamics.

Each concrete [`EquationOfMotion`][jbubble.bubble.eom.EquationOfMotion]
assembles a [`GasModel`][jbubble.bubble.gas.GasModel], a
[`ShellModel`][jbubble.bubble.shell.ShellModel], and a
[`MediumModel`][jbubble.bubble.medium.MediumModel] into a complete ODE
right-hand side that returns the time derivative as a
[`BubbleState`][jbubble.bubble.state.BubbleState].
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .gas import GasModel
from .medium import MediumModel
from .shell import ShellModel
from .state import BubbleState

__all__ = [
    "EquationOfMotion",
    "RayleighPlesset",
    "ModifiedRayleighPlesset",
    "KellerMiksis",
    "Gilmore",
]


class EquationOfMotion[StateType: BubbleState](eqx.Module, abc.ABC):
    r"""Macroscopic equation of motion for bubble dynamics.

    Assembles a [`GasModel`][jbubble.bubble.gas.GasModel], a
    [`ShellModel`][jbubble.bubble.shell.ShellModel], and a
    [`MediumModel`][jbubble.bubble.medium.MediumModel] into a complete ODE
    right-hand side. Concrete subclasses encode different formulations
    (Rayleigh-Plesset, Keller-Miksis, and others), which differ in how they
    relate the liquid-side boundary pressure $p_L$ to the radial
    acceleration $\ddot{R}$.

    The type parameter `StateType` binds each equation of motion (EoM) to
    the state it operates on. The built-in EoMs use
    [`BubbleState`][jbubble.bubble.state.BubbleState]; an EoM that carries
    extra degrees of freedom binds a `BubbleState` subclass, so the type
    checker catches a wrong state type before it fails at runtime deep
    inside a JAX trace.

    The concrete helper [`p_L`][jbubble.bubble.eom.EquationOfMotion.p_L]
    computes the liquid-side boundary pressure:

    $$
    p_L = p_\text{gas}(\text{state}) - p_\text{shell}(\text{state})
        - p_\text{medium}(\text{state})
    $$

    Because `p_L` is a regular method on an Equinox module, JAX
    differentiates through it automatically. EoMs that need
    $\mathrm{d}p_L/\mathrm{d}t$, such as Keller-Miksis, use
    `jax.grad(self.p_L)` instead of hand-coding analytical derivatives for
    every combination of components.

    The EoM receives the driving acoustic pressure as a callable `p_ac_fn`,
    so EoMs that need $\mathrm{d}p_\text{ac}/\mathrm{d}t$ compute it with
    `jax.grad`.

    [`initial_state`][jbubble.bubble.eom.EquationOfMotion.initial_state]
    seeds the equilibrium configuration, `R0` and `P_gas0`, into the state.
    The ODE carries them as frozen constants, with zero time derivatives in
    the standard case.

    [`state_scale`][jbubble.bubble.eom.EquationOfMotion.state_scale] gives
    the magnitude of each state field, which
    [`solve_eom`][jbubble.solver.solve_eom] uses to make the solver
    tolerances dimensionless. Override it in a subclass that adds state
    fields.

    Parameters
    ----------
    gas : GasModel
        Internal gas pressure model.
    shell : ShellModel
        Shell or coating model.
    medium : MediumModel
        Surrounding medium model.
    R0 : float or jax.Array
        Equilibrium bubble radius [m].
    P_amb : float or jax.Array
        Ambient (far-field) pressure [Pa].
    rho_L : float or jax.Array
        Liquid density [kg/m³].
    """

    gas: GasModel
    shell: ShellModel
    medium: MediumModel

    R0: ArrayLike
    P_amb: ArrayLike
    rho_L: ArrayLike

    def p_L(self, state: BubbleState) -> jax.Array:
        r"""Liquid-side boundary pressure.

        $$
        p_L = p_\text{gas}(\text{state}) - p_\text{shell}(\text{state})
            - p_\text{medium}(\text{state})
        $$

        Parameters
        ----------
        state : BubbleState
            Current bubble state.

        Returns
        -------
        jax.Array
            Boundary pressure [Pa].
        """
        return self.gas(state) - self.shell(state) - self.medium(state)

    def initial_state(
        self,
        *,
        R: ArrayLike | None = None,
        R_dot: ArrayLike | None = None,
    ) -> BubbleState:
        r"""Return an initial state seeded with the equilibrium configuration.

        By default the bubble starts at rest at its equilibrium radius. Seeds
        `R0` and the Laplace-equilibrium gas pressure into the state:

        $$
        P_{\text{gas},0} = P_\text{amb} + \frac{2\sigma(R_0)}{R_0}
        $$

        Pass `R` or `R_dot` to start away from equilibrium, for example
        `eom.initial_state(R=1.2 * eom.R0)` for a bubble released from rest
        at 1.2 times its equilibrium radius. `R0` and `P_gas0` stay at their
        equilibrium values, so the gas pressure at `R` follows the gas law.

        Override this method for coupled systems with a larger state vector.
        An override must accept the same keyword arguments.

        Parameters
        ----------
        R : float or jax.Array, optional
            Initial radius [m]. `None` uses `R0`.
        R_dot : float or jax.Array, optional
            Initial wall velocity [m/s]. `None` uses `0`.

        Returns
        -------
        BubbleState
            State with `R`, `R_dot`, `R0`, and `P_gas0` set.
        """
        R0 = jnp.asarray(self.R0)
        P_gas0 = (
            jnp.asarray(self.P_amb)
            + 2.0 * self.shell.sigma(BubbleState(R=R0, R0=R0)) / R0
        )
        R_init = R0 if R is None else jnp.asarray(R)
        R_dot_init = jnp.zeros_like(R0) if R_dot is None else jnp.asarray(R_dot)
        return BubbleState(R=R_init, R_dot=R_dot_init, R0=R0, P_gas0=P_gas0)

    def state_scale(self, state: StateType) -> StateType:
        r"""Return the characteristic magnitude of each state field.

        [`solve_eom`][jbubble.solver.solve_eom] integrates the dimensionless
        state `state / state_scale(state)`, so the step-size controller's
        `rtol` and `atol` mean the same thing for a 50 nm bubble as for a
        50 µm one. The default scales are:

        - `R` and `R0`: the equilibrium radius `state.R0` [m].
        - `R_dot`: the velocity scale $\sqrt{P_\text{amb}/\rho_L}$
          [m/s], about 10 m/s in water at atmospheric pressure. It's the
          natural velocity of bubble dynamics: $R_0\omega_0 \approx
          \sqrt{3\kappa P_\text{amb}/\rho_L}$ for the Minnaert frequency
          $\omega_0$.
        - `P_gas0`: its own value `state.P_gas0` [Pa], or `P_amb` if that
          value isn't positive.
        - Any other field of a `BubbleState` subclass: `1`, that is, SI
          units.

        Because `R0` and `P_gas0` are scaled by their own values, the solver
        sees them exactly: `(x / x) * x == x` in floating point.

        Override this method to give extra state fields a scale. The scale
        must be positive and finite; the solver treats it as a constant.

        Parameters
        ----------
        state : StateType
            Initial state, with a positive `R0`.

        Returns
        -------
        StateType
            Scale for each field, with the same structure as `state`.
        """
        P_amb = jnp.asarray(self.P_amb)
        velocity = jnp.sqrt(P_amb / jnp.asarray(self.rho_L))
        R0 = jnp.asarray(state.R0)
        P_gas0 = jnp.asarray(state.P_gas0)
        pressure = jnp.where(P_gas0 > 0, P_gas0, P_amb)
        ones = jax.tree_util.tree_map(jnp.ones_like, state)
        values = (R0, velocity, R0, pressure)
        return eqx.tree_at(
            lambda s: (s.R, s.R_dot, s.R0, s.P_gas0),
            ones,
            tuple(
                jnp.broadcast_to(v, jnp.shape(leaf)).astype(jnp.result_type(leaf))
                for v, leaf in zip(
                    values, (state.R, state.R_dot, state.R0, state.P_gas0), strict=True
                )
            ),
        )

    @abc.abstractmethod
    def __call__(
        self,
        t: Any,
        state: StateType,
        p_ac_fn: Callable,
    ) -> StateType:
        r"""Compute the ODE right-hand side, $\mathrm{d}(\text{state})/\mathrm{d}t$.

        Parameters
        ----------
        t : float or jax.Array
            Current time [s].
        state : StateType
            Current bubble state. Its type matches the EoM's state parameter.
        p_ac_fn : callable
            Driving acoustic pressure [Pa] as a function of time,
            `t -> scalar`.

        Returns
        -------
        StateType
            Time derivative of the state. The `R0` and `P_gas0` derivatives
            are zero in the standard case.
        """
        ...


class RayleighPlesset(EquationOfMotion[BubbleState]):
    r"""Rayleigh-Plesset equation of motion (incompressible liquid).

    $$
    R\ddot{R} + \frac{3}{2}\dot{R}^2
        = \frac{1}{\rho_L}\left(p_L - P_\text{amb} - p_\text{ac}\right)
    $$

    The simplest bubble dynamics EoM, which assumes an incompressible
    surrounding liquid. It takes only the parameters of
    [`EquationOfMotion`][jbubble.bubble.eom.EquationOfMotion].
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
    r"""Modified Rayleigh-Plesset equation with gas radiation damping.

    Adds a first-order compressibility correction to the gas pressure
    term only, as used by Marmottant et al. (2005):

    $$
    R\ddot{R} + \frac{3}{2}\dot{R}^2
        = \frac{1}{\rho_L}\left(p_L + \frac{R}{c_L}\frac{\mathrm{d}p_\text{gas}}{\mathrm{d}t}
        - P_\text{amb} - p_\text{ac}\right)
    $$

    where autodiff computes
    $\mathrm{d}p_\text{gas}/\mathrm{d}t = (\partial p_\text{gas}/\partial R)\,\dot{R}$.

    It takes the parameters of
    [`EquationOfMotion`][jbubble.bubble.eom.EquationOfMotion] plus `c_L`.

    Parameters
    ----------
    c_L : float or jax.Array
        Speed of sound in the liquid [m/s].
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
    r"""Keller-Miksis equation of motion (first-order compressibility).

    Accounts for liquid compressibility up to first order in the Mach
    number $M = \dot{R}/c_L$:

    $$
    (1 - M) R\ddot{R} + \frac{3}{2}\left(1 - \frac{M}{3}\right)\dot{R}^2
        = \frac{1 + M}{\rho_L}\left(p_L - P_\text{amb} - p_\text{ac}\right)
        + \frac{R}{\rho_L c_L}\left(\frac{\mathrm{d}p_L}{\mathrm{d}t}
        - \frac{\mathrm{d}p_\text{ac}}{\mathrm{d}t}\right)
    $$

    JAX autodiff computes the time derivative
    $\mathrm{d}p_L/\mathrm{d}t$ **automatically**, through the chain rule
    on `p_L`:

    $$
    \frac{\mathrm{d}p_L}{\mathrm{d}t}
        = \frac{\partial p_L}{\partial R}\dot{R}
        + \frac{\partial p_L}{\partial \dot{R}}\ddot{R}
    $$

    The $\ddot{R}$ term moves to the left-hand side, so this EoM works with
    any combination of gas, shell, and medium models without hand-coded
    derivatives.

    It takes the parameters of
    [`EquationOfMotion`][jbubble.bubble.eom.EquationOfMotion] plus `c_L`.

    Parameters
    ----------
    c_L : float or jax.Array
        Speed of sound in the liquid [m/s].
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
    r"""Gilmore equation of motion (Kirkwood-Bethe hypothesis).

    Improves on Keller-Miksis by treating liquid compressibility through
    the Tait equation of state rather than a linear approximation. The
    enthalpy $H$ and local speed of sound $C$ at the bubble wall are exact
    functions of the wall pressure, which gives accurate results at high
    Mach numbers:

    $$
    \left(1 - \frac{\dot{R}}{C}\right) R\ddot{R}
        + \frac{3}{2}\left(1 - \frac{\dot{R}}{3C}\right)\dot{R}^2
        = \left(1 + \frac{\dot{R}}{C}\right) H
        + \frac{R}{C}\left(1 - \frac{\dot{R}}{C}\right)\dot{H}
    $$

    where the enthalpy difference between the bubble wall and the far field
    is

    $$
    \begin{aligned}
    H &= \frac{n}{n-1} K \left[(p_L + B)^{(n-1)/n}
        - (p_\infty + B)^{(n-1)/n}\right], \\
    K &= \frac{(P_\text{amb} + B)^{1/n}}{\rho_L},
    \end{aligned}
    $$

    and the local sound speed satisfies

    $$
    C^2 = n K (p_\infty + B)^{(n-1)/n} + (n - 1) H,
    $$

    with $p_\infty = P_\text{amb} + p_\text{ac}$. The code clamps $C^2$ to
    at least 1 m²/s² before it takes the square root.

    The chain rule expands $\dot{H}$:

    $$
    \dot{H} = \frac{\partial H}{\partial p_L}\frac{\mathrm{d}p_L}{\mathrm{d}t}
        + \frac{\partial H}{\partial p_\infty}\frac{\mathrm{d}p_\text{ac}}{\mathrm{d}t},
    \qquad
    \frac{\partial H}{\partial p_L} = K (p_L + B)^{-1/n},
    \qquad
    \frac{\partial H}{\partial p_\infty} = -K (p_\infty + B)^{-1/n}.
    $$

    The partial derivatives of $H$ come from the Tait formula.
    `jax.grad(self.p_L)` gives $\partial p_L/\partial R$ and
    $\partial p_L/\partial \dot{R}$, from which the chain rule builds
    $\mathrm{d}p_L/\mathrm{d}t$; `jax.grad(p_ac_fn)` gives
    $\mathrm{d}p_\text{ac}/\mathrm{d}t$. The $\ddot{R}$ coupling through
    $\partial p_L/\partial \dot{R}$ moves into the denominator, in the
    same way as in [`KellerMiksis`][jbubble.bubble.eom.KellerMiksis].

    The default Tait parameters for water, $n = 7.15$ and
    $B = 3.046 \times 10^8$ Pa, are those of Gümmer, Schenke & Denner
    (2021). Gilmore (1952) quotes the rounder "$B \approx 3000$ atm and
    $n \approx 7$". The Tait parameters fix the far-field sound speed of the
    liquid at rest,

    $$
    c_\infty = \sqrt{\frac{n\,(P_\text{amb} + B)}{\rho_L}},
    $$

    which is 1477 m/s for the defaults with $P_\text{amb} = 101\,325$ Pa and
    $\rho_L = 998$ kg/m³. This EoM has no `c_L` field: to compare it with
    [`KellerMiksis`][jbubble.bubble.eom.KellerMiksis], give the Keller-Miksis
    model $c_L = c_\infty$, or choose `B_tait` so that $c_\infty$ matches
    your `c_L`.

    It takes the parameters of
    [`EquationOfMotion`][jbubble.bubble.eom.EquationOfMotion] plus
    `n_tait` and `B_tait`.

    Parameters
    ----------
    n_tait : float or jax.Array
        Tait exponent (dimensionless). Default: `7.15`.
    B_tait : float or jax.Array
        Tait pressure constant [Pa]. Default: `3.046e8`.

    References
    ----------
    Gilmore, F. R. (1952). *The growth or collapse of a spherical bubble
    in a viscous compressible liquid.* Hydrodynamics Laboratory Report
    26-4, California Institute of Technology.

    Gümmer, J., Schenke, S., & Denner, F. (2021). Modelling lipid-coated
    microbubbles in focused ultrasound applications at subresonance
    frequencies. *Ultrasound in Medicine & Biology*, 47(10), 2958-2979,
    Eqs. 2-6. <https://doi.org/10.1016/j.ultrasmedbio.2021.06.012>
    """

    n_tait: ArrayLike = 7.15
    B_tait: ArrayLike = 3.046e8

    def _tait_K(self) -> jax.Array:
        r"""Return the Tait prefactor, $K = (P_\text{amb} + B)^{1/n} / \rho_L$."""
        return jnp.asarray(
            (self.P_amb + self.B_tait) ** (1.0 / self.n_tait) / self.rho_L
        )

    def _H_and_C(self, p_L: jax.Array, p_inf: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the enthalpy $H$ [m²/s²] and bubble-wall sound speed $C$ [m/s]."""
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
