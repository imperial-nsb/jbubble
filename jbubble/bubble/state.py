"""Bubble state PyTrees for ODE integration.

``BubbleState`` is the standard state vector for a single unconfined bubble.
``ConfinedBubbleState`` extends it with a vessel wall degree of freedom.

Using Equinox modules as ODE states guarantees strict PyTree congruency
with diffrax and enables painless multi-physics extensions (thermal
dynamics, rectified diffusion, etc.) by adding new fields.

The equilibrium fields ``R0`` and ``P_gas0`` are carried alongside the
dynamic variables so that all gas and shell models can read them from the
state without requiring separate storage or extra arguments.  In the
standard case their time derivatives are zero (frozen constants).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class BubbleState(eqx.Module):
    """Standard bubble state.

    Fields
    ------
    R : jax.Array
        Bubble wall radius  [m].
    R_dot : jax.Array
        Bubble wall velocity  [m/s].
    R0 : jax.Array
        Equilibrium bubble radius  [m].  Frozen (dR0/dt = 0) in the
        standard case; becomes a slow state variable for rectified
        diffusion etc.
    P_gas0 : jax.Array
        Equilibrium gas pressure  [Pa].  Frozen in the standard case.
    """

    R: jax.Array
    R_dot: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(()),
        kw_only=True,
    )
    R0: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(()),
        kw_only=True,
    )
    P_gas0: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(()),
        kw_only=True,
    )


class ThermalBubbleState(BubbleState):
    """Bubble state with a gas-temperature degree of freedom.

    Extends :class:`BubbleState` with the (spatially lumped) gas temperature
    ``T`` and its equilibrium value ``T0``.  Used by gas models that resolve
    the gas thermodynamics in time — e.g. :class:`~jbubble.bubble.gas.ThermalGas`,
    whose energy equation makes ``T`` a genuine dynamic variable and thereby
    introduces *thermal damping* (irreversible heat exchange with the liquid).

    ``T0`` is carried alongside ``R0``/``P_gas0`` as a frozen equilibrium
    constant (dT0/dt = 0), so gas models can read it from the state without
    extra plumbing, mirroring the existing equilibrium fields.

    Fields
    ------
    T : jax.Array
        Gas temperature  [K].
    T0 : jax.Array
        Equilibrium gas temperature  [K].  Frozen (dT0/dt = 0).
    """

    T: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(()),
        kw_only=True,
    )
    T0: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(()),
        kw_only=True,
    )


class HystereticShellState(BubbleState):
    """Bubble state with a shell-integrity degree of freedom.

    Extends :class:`BubbleState` with a dimensionless shell integrity ``phi``
    (1 = intact monolayer, 0 = fully ruptured / shed).  Used by shell models
    whose surface tension depends on loading *history* — e.g.
    :class:`~jbubble.bubble.shell.HystereticShell`, where ``phi`` drops
    irreversibly on over-expansion and heals back on compression, so the σ–R
    curve encloses area and dissipates energy (shell hysteresis) rather than
    being a single-valued, reversible function of R.

    Fields
    ------
    phi : jax.Array
        Shell integrity  (dimensionless, 0–1).  Initialised to 1 (intact).
    """

    phi: jax.Array = eqx.field(
        default_factory=lambda: jnp.ones(()),
        kw_only=True,
    )


class ConfinedBubbleState(BubbleState):
    """State for a bubble confined in an elastic spherical vessel.

    Extends ``BubbleState`` with the vessel wall radius and velocity.

    Fields
    ------
    a : jax.Array
        Vessel wall radius  [m].
    a_dot : jax.Array
        Vessel wall velocity  [m/s].
    """

    a: jax.Array
    a_dot: jax.Array
