"""Tests for jbubble.bubble.state."""

import jax
import jax.numpy as jnp

from jbubble.bubble.state import BubbleState, ConfinedBubbleState


class TestBubbleState:
    def test_create_with_R_only(self):
        s = BubbleState(R=jnp.asarray(2e-6))
        assert float(s.R) == 2e-6
        assert float(s.R_dot) == 0.0
        assert float(s.R0) == 0.0
        assert float(s.P_gas0) == 0.0

    def test_create_with_all_fields(self):
        s = BubbleState(
            R=jnp.asarray(2e-6),
            R_dot=jnp.asarray(1.0),
            R0=jnp.asarray(2e-6),
            P_gas0=jnp.asarray(1e5),
        )
        assert float(s.R_dot) == 1.0
        assert float(s.R0) == 2e-6
        assert float(s.P_gas0) == 1e5

    def test_is_pytree(self):
        s = BubbleState(R=jnp.asarray(2e-6), R0=jnp.asarray(2e-6))
        leaves = jax.tree_util.tree_leaves(s)
        assert all(isinstance(l, jax.Array) for l in leaves)

    def test_jit_compatible(self):
        @jax.jit
        def get_R(state):
            return state.R

        s = BubbleState(R=jnp.asarray(2e-6))
        assert float(get_R(s)) == 2e-6

    def test_vmap_compatible(self):
        Rs = jnp.array([1e-6, 2e-6, 3e-6])
        states = BubbleState(
            R=Rs,
            R_dot=jnp.zeros(3),
            R0=Rs,
            P_gas0=jnp.ones(3) * 1e5,
        )

        @jax.vmap
        def get_ratio(s):
            return s.R / s.R0

        result = get_ratio(states)
        assert jnp.allclose(result, jnp.ones(3))


class TestConfinedBubbleState:
    def test_create(self):
        s = ConfinedBubbleState(
            R=jnp.asarray(2e-6),
            a=jnp.asarray(50e-6),
            a_dot=jnp.asarray(0.0),
        )
        assert float(s.R) == 2e-6
        assert float(s.a) == 50e-6
        assert float(s.a_dot) == 0.0

    def test_inherits_from_bubble_state(self):
        s = ConfinedBubbleState(
            R=jnp.asarray(2e-6),
            a=jnp.asarray(50e-6),
            a_dot=jnp.asarray(0.0),
        )
        assert isinstance(s, BubbleState)

    def test_has_bubble_state_fields(self):
        s = ConfinedBubbleState(
            R=jnp.asarray(2e-6),
            R_dot=jnp.asarray(0.5),
            R0=jnp.asarray(2e-6),
            P_gas0=jnp.asarray(1e5),
            a=jnp.asarray(50e-6),
            a_dot=jnp.asarray(0.1),
        )
        assert float(s.R_dot) == 0.5
        assert float(s.P_gas0) == 1e5
