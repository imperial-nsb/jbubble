"""Gas pressure models.

Computes the outward gas pressure p_gas as a function of the
instantaneous bubble state.  All models read ``state.R0`` and
``state.P_gas0`` directly from the state, so they need no separate
storage of equilibrium parameters.  The ``EquationOfMotion`` seeds
these fields in ``initial_state()`` and the ODE carries them forward
(frozen at zero derivative in the standard case).
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .property import Property, as_property
from .state import BubbleState, ThermalBubbleState


class GasModel(eqx.Module, abc.ABC):
    """Internal gas pressure model.

    Computes the outward gas pressure p_gas as a function of the
    instantaneous state.  The equilibrium configuration (``R0``,
    ``P_gas0``) is read directly from ``state``, so gas models only
    store their intrinsic physical parameters (e.g. the polytropic
    exponent).

    Examples: polytropic law, van der Waals corrected gas.
    """

    @abc.abstractmethod
    def __call__(self, state: BubbleState) -> jax.Array:
        """Compute gas pressure p_gas(state).

        Parameters
        ----------
        state : BubbleState
            Current bubble state.  Uses ``state.R``, ``state.R0``,
            and ``state.P_gas0``.

        Returns
        -------
        scalar
            Gas pressure.
        """
        ...

    # ── optional multi-physics hooks (default: no extra state) ───────────────
    # A gas model with its own dynamics (e.g. a resolved temperature) overrides
    # these two methods; the EoM calls them generically so no EoM code needs to
    # know which gas model is in use.  The defaults make a plain gas model add
    # nothing, so existing behaviour is unchanged.

    def augment_initial_state(self, base: BubbleState) -> BubbleState:
        """Return the initial state, extended with any gas-owned DOFs.

        Called by ``EquationOfMotion.initial_state``.  Default: pass through.
        """
        return base

    def d_state(self, state: BubbleState) -> BubbleState:
        """Gas-owned contribution to d(state)/dt.

        Returns a state-shaped PyTree that is zero everywhere except the gas
        model's own dynamic fields (e.g. dT/dt).  The EoM adds this into the
        assembled derivative, so the gas dynamics ride alongside the wall
        dynamics.  Default: all zeros (a purely algebraic gas law).
        """
        return jax.tree_util.tree_map(jnp.zeros_like, state)


class PolytropicGas(GasModel):
    """Polytropic gas law.

    p_gas(R) = P_gas0 (R0 / R)^(3 gamma)

    Fields
    ------
    gamma : float
        Polytropic exponent  (1.0 = isothermal, 1.4 = adiabatic air).
    """

    gamma: Property = eqx.field(converter=as_property)

    def __call__(self, state: BubbleState) -> jax.Array:
        return state.P_gas0 * (state.R0 / state.R) ** (3.0 * self.gamma(state))


class VanDerWaalsGas(GasModel):
    """Hard-core corrected polytropic gas (van der Waals).

    p_gas(R) = P_gas0 ((R0^3 - h^3) / (R^3 - h^3))^gamma

    where *h = h_frac * R0* is the van der Waals hard-core radius.

    Fields
    ------
    gamma : float
        Polytropic exponent.
    h_frac : float
        Hard-core radius as a fraction of R0  (dimensionless).
        A common value for lipid shells is 1/5.61 ≈ 0.178.
    """

    gamma: Property = eqx.field(converter=as_property)
    h_frac: Property = eqx.field(converter=as_property)

    def __call__(self, state: BubbleState) -> jax.Array:
        h = self.h_frac(state) * state.R0
        return state.P_gas0 * (
            (state.R0**3 - h**3) / (state.R**3 - h**3)
        ) ** self.gamma(state)


class DiffusiveGas(PolytropicGas):
    """Polytropic gas that slowly loses gas to the liquid (gas diffusion).

    A driven bubble exchanges gas with the surrounding liquid; under these
    conditions the *net* flux is outward (dissolution / shedding), so the bubble
    permanently shrinks — its equilibrium radius ``R0`` drifts down over the
    pulse and the oscillation re-centres lower.  A standard model holds ``R0``
    fixed and cannot reproduce that baseline drift.

    This model makes ``R0`` a **slow state variable** (exactly as anticipated in
    ``state.BubbleState``) driven by the oscillation activity::

        Ṙ0 = − k_diff · s² · R0 ,    s = (R − R0) / R0

    so only oscillating bubbles lose gas and larger oscillations lose it faster
    (an amplitude-driven, monotone shrinkage; s² ≥ 0 ⇒ Ṙ0 ≤ 0).  The pressure
    law is the usual polytropic ``P_gas0 (R0/R)^{3γ}`` evaluated at the current
    (shrinking) ``R0``.  Because diffusion is slow compared with the oscillation,
    this is the standard quasi-static treatment.

    The equilibrium gas pressure ``P_gas0`` is tracked from the Laplace balance
    ``P_gas0 = P_amb + 2 σ_lap / R0`` as ``R0`` moves::

        Ṗ_gas0 = − (2 σ_lap / R0²) · Ṙ0

    With the default ``sigma_lap = 0`` this term vanishes and ``P_gas0`` stays
    frozen (the leading-order model — only the resting radius drifts); set it to
    the rest surface tension for a self-consistent pressure.

    No new state class is needed: ``R0`` and ``P_gas0`` are already carried by
    :class:`~jbubble.bubble.state.BubbleState`; this model simply gives them a
    non-zero, slow derivative.

    Fields
    ------
    gamma : float or Property
        Polytropic exponent  (inherited).
    k_diff : float or Property
        Gas-loss rate constant  [1/s]  (per unit squared strain).
    sigma_lap : float
        Surface tension used in the P_gas0 Laplace tracking  [N/m].  Default 0.
    """

    k_diff: Property = eqx.field(converter=as_property)
    sigma_lap: float = 0.0

    def d_state(self, state: BubbleState) -> BubbleState:
        s = (state.R - state.R0) / state.R0
        dR0 = -self.k_diff(state) * s * s * state.R0
        dP0 = -2.0 * self.sigma_lap / state.R0**2 * dR0
        zero = jax.tree_util.tree_map(jnp.zeros_like, state)
        return eqx.tree_at(lambda st: (st.R0, st.P_gas0), zero, (dR0, dP0))


class ThermalGas(GasModel):
    """Lumped-thermal gas with an explicit temperature state (thermal damping).

    A polytropic law is *reversible*: it stores no heat and so dissipates no
    energy, leaving the free ring-down of a bubble under-damped.  Real bubbles
    lose energy through heat conducted between the compressing gas and the
    liquid.  This model resolves that by carrying the (spatially uniform) gas
    temperature ``T`` as a dynamic state variable on
    :class:`~jbubble.bubble.state.ThermalBubbleState`.

    Ideal gas at uniform temperature, ``pV = N k_B T``, gives the pressure::

        p_gas = P_gas0 · (R0 / R)^3 · (T / T0)

    and the first law (``N C_v Ṫ = −p V̇ + Q̇``) with Newtonian heat exchange
    toward the wall temperature ``T0`` gives the temperature evolution::

        Ṫ = −3 (γ − 1) T Ṙ / R  −  (T − T0) / τ_th

    The first term is reversible adiabatic heating/cooling; the second is the
    irreversible relaxation that dissipates energy.  Limits:

    * ``τ_th → ∞`` (no heat exchange): ``T = T0 (R0/R)^{3(γ−1)}`` so
      ``p_gas = P_gas0 (R0/R)^{3γ}`` — recovers the **adiabatic** polytropic law.
    * ``τ_th → 0`` (instant exchange): ``T → T0`` so
      ``p_gas = P_gas0 (R0/R)^3`` — recovers the **isothermal** law (γ = 1).
    * intermediate ``τ_th`` (``ω τ_th ~ 1``): maximal thermal damping.

    Here ``γ`` is the true adiabatic ratio of specific heats of the gas (not an
    effective polytropic exponent — the effective behaviour now emerges from the
    dynamics), and ``τ_th`` the thermal relaxation time (≈ R0²/α for gas thermal
    diffusivity α; left as a free parameter since it also absorbs the lumped
    approximation).

    Fields
    ------
    gamma : float or Property
        Adiabatic ratio of specific heats  (dimensionless, > 1).
    thermal_time : float or Property
        Thermal relaxation time τ_th  [s].
    T_amb : float
        Equilibrium / ambient gas temperature T0  [K].  Default 293.0.
    """

    gamma: Property = eqx.field(converter=as_property)
    thermal_time: Property = eqx.field(converter=as_property)
    T_amb: float = 293.0

    def __call__(self, state: BubbleState) -> jax.Array:
        # p = P_gas0 (R0/R)^3 (T/T0); reads T, T0 from the thermal state.
        return (
            state.P_gas0
            * (state.R0 / state.R) ** 3
            * (state.T / state.T0)  # ty: ignore[unresolved-attribute]
        )

    def _dT_dt(self, state: BubbleState) -> jax.Array:
        gamma = self.gamma(state)
        tau = self.thermal_time(state)
        T = state.T  # ty: ignore[unresolved-attribute]
        T0 = state.T0  # ty: ignore[unresolved-attribute]
        adiabatic = -3.0 * (gamma - 1.0) * T * state.R_dot / state.R
        relaxation = -(T - T0) / tau
        return adiabatic + relaxation

    def augment_initial_state(self, base: BubbleState) -> ThermalBubbleState:
        T0 = jnp.asarray(self.T_amb, dtype=base.R.dtype)
        return ThermalBubbleState(
            R=base.R,
            R_dot=base.R_dot,
            R0=base.R0,
            P_gas0=base.P_gas0,
            T=T0,
            T0=T0,
        )

    def d_state(self, state: BubbleState) -> BubbleState:
        zero = jax.tree_util.tree_map(jnp.zeros_like, state)
        return eqx.tree_at(
            lambda s: s.T,  # ty: ignore[unresolved-attribute]
            zero,
            self._dT_dt(state),
        )
