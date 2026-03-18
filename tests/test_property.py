"""Tests for jbubble.bubble.property."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jbubble.bubble.property import ConstantProperty, NeuralProperty, Property, as_property
from jbubble.bubble.state import BubbleState


@pytest.fixture
def state():
    return BubbleState(
        R=jnp.asarray(3e-6),
        R_dot=jnp.asarray(0.1),
        R0=jnp.asarray(2e-6),
        P_gas0=jnp.asarray(1e5),
    )


class TestConstantProperty:
    def test_returns_constant_value(self, state):
        p = ConstantProperty(val=0.072)
        assert float(p(state)) == pytest.approx(0.072, rel=1e-6)

    def test_independent_of_state(self, state):
        p = ConstantProperty(val=42.0)
        s2 = BubbleState(R=jnp.asarray(1e-6), R0=jnp.asarray(1e-6))
        assert float(p(state)) == pytest.approx(float(p(s2)), rel=1e-6)

    def test_is_property_subclass(self):
        p = ConstantProperty(val=1.0)
        assert isinstance(p, Property)

    def test_grad_through_constant(self, state):
        p = ConstantProperty(val=0.072)
        grad = jax.grad(lambda s: p(s).sum())(state)
        # gradient of a constant w.r.t. R should be zero (up to the R*0.0 trick)
        assert float(grad.R) == pytest.approx(0.0, abs=1e-12)


class TestAsProperty:
    def test_float_to_constant_property(self):
        result = as_property(0.072)
        assert isinstance(result, ConstantProperty)
        assert float(result.val) == 0.072

    def test_int_to_constant_property(self):
        result = as_property(7)
        assert isinstance(result, ConstantProperty)

    def test_property_passthrough(self):
        p = ConstantProperty(val=0.072)
        result = as_property(p)
        assert result is p

    def test_jnp_array(self):
        result = as_property(jnp.asarray(0.072))
        assert isinstance(result, ConstantProperty)


class TestNeuralProperty:
    def test_creates_and_evaluates(self):
        key = jax.random.PRNGKey(0)
        mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
        np = NeuralProperty(net=mlp)
        state = BubbleState(
            R=jnp.asarray(3e-6),
            R0=jnp.asarray(2e-6),
            P_gas0=jnp.asarray(1e5),
        )
        result = np(state)
        assert result.shape == ()

    def test_is_differentiable(self):
        key = jax.random.PRNGKey(42)
        mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
        np_prop = NeuralProperty(net=mlp)
        state = BubbleState(
            R=jnp.asarray(3e-6),
            R0=jnp.asarray(2e-6),
            P_gas0=jnp.asarray(1e5),
        )
        grad = jax.grad(lambda s: np_prop(s))(state)
        # Gradient w.r.t. R should be nonzero in general
        assert jnp.isfinite(grad.R)

    def test_normalises_input(self):
        """NeuralProperty feeds R/R0 to the network."""
        key = jax.random.PRNGKey(1)
        # Identity network approximation: just checks input is R/R0
        mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=4, depth=1, key=key)
        np_prop = NeuralProperty(net=mlp)
        s1 = BubbleState(R=jnp.asarray(2e-6), R0=jnp.asarray(2e-6))
        s2 = BubbleState(R=jnp.asarray(4e-6), R0=jnp.asarray(4e-6))
        # Same R/R0 ratio → same output
        assert float(np_prop(s1)) == pytest.approx(float(np_prop(s2)), rel=1e-5)
