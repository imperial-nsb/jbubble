"""Universal PyTree state for bubble dynamics.

Provides ``BubbleState`` as the standard 2-DOF state and
``ConfinedBubbleState`` for models with an additional vessel wall
(e.g. ``SphericalConfinement``).

Using an Equinox module as the ODE state guarantees strict PyTree
congruency with diffrax and enables painless future multi-physics
extensions (thermal dynamics, vapor mass, etc.).
"""

import equinox as eqx
import jax


class BubbleState(eqx.Module):
    """Standard 2-DOF bubble state.

    Fields
    ------
    R : jax.Array
        Bubble wall radius  [m].
    R_dot : jax.Array
        Bubble wall velocity  [m/s].
    """

    R: jax.Array
    R_dot: jax.Array


class ConfinedBubbleState(BubbleState):
    """4-DOF state for a bubble confined in an elastic vessel.

    Extends ``BubbleState`` with the vessel wall radius and velocity.

    Fields
    ------
    R : jax.Array
        Bubble wall radius  [m].
    R_dot : jax.Array
        Bubble wall velocity  [m/s].
    a : jax.Array
        Vessel wall radius  [m].
    a_dot : jax.Array
        Vessel wall velocity  [m/s].
    """

    a: jax.Array
    a_dot: jax.Array
