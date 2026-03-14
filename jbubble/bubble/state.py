"""Universal PyTree state for bubble dynamics.

Provides ``BubbleState`` as the standard state and
``ConfinedBubbleState`` for models with an additional vessel wall
(e.g. ``SphericalConfinement``).

Using an Equinox module as the ODE state guarantees strict PyTree
congruency with diffrax and enables painless future multi-physics
extensions (thermal dynamics, rectified diffusion, etc.).

The ``R0`` and ``P_gas0`` fields carry the equilibrium configuration
alongside the dynamic variables.  In the standard case their time
derivatives are zero (frozen constants).  For extensions such as
rectified diffusion, ``R0`` can be given a non-trivial ODE without
any changes to the gas models — they simply read ``state.R0`` and
``state.P_gas0`` as before.
"""

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
