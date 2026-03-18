"""Tests for jbubble.pulse.envelope."""

import jax
import jax.numpy as jnp
import pytest

from jbubble.pulse.envelope import (
    HannEnvelope,
    RectangularEnvelope,
    SoftRectangularEnvelope,
    TukeyEnvelope,
)


DURATION = 5e-6


class TestRectangularEnvelope:
    def test_one_inside(self):
        env = RectangularEnvelope()
        assert float(env(jnp.asarray(2.5e-6), DURATION)) == 1.0

    def test_zero_before(self):
        env = RectangularEnvelope()
        assert float(env(jnp.asarray(-1e-6), DURATION)) == 0.0

    def test_zero_after(self):
        env = RectangularEnvelope()
        assert float(env(jnp.asarray(6e-6), DURATION)) == 0.0

    def test_one_at_boundaries(self):
        env = RectangularEnvelope()
        assert float(env(jnp.asarray(0.0), DURATION)) == 1.0
        assert float(env(jnp.asarray(DURATION), DURATION)) == 1.0


class TestHannEnvelope:
    def test_zero_at_boundaries(self):
        env = HannEnvelope()
        assert float(env(jnp.asarray(0.0), DURATION)) == pytest.approx(0.0, abs=1e-10)
        assert float(env(jnp.asarray(DURATION), DURATION)) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_one_at_center(self):
        env = HannEnvelope()
        assert float(env(jnp.asarray(DURATION / 2), DURATION)) == pytest.approx(
            1.0, abs=1e-10
        )

    def test_zero_outside(self):
        env = HannEnvelope()
        assert float(env(jnp.asarray(-1e-6), DURATION)) == 0.0
        assert float(env(jnp.asarray(10e-6), DURATION)) == 0.0

    def test_symmetric(self):
        env = HannEnvelope()
        val_left = float(env(jnp.asarray(DURATION * 0.25), DURATION))
        val_right = float(env(jnp.asarray(DURATION * 0.75), DURATION))
        assert val_left == pytest.approx(val_right, rel=1e-10)


class TestSoftRectangularEnvelope:
    def test_near_one_in_middle(self):
        env = SoftRectangularEnvelope(steepness=100.0)
        val = float(env(jnp.asarray(DURATION / 2), DURATION))
        assert val == pytest.approx(1.0, abs=0.01)

    def test_near_zero_outside(self):
        env = SoftRectangularEnvelope(steepness=100.0)
        before = float(env(jnp.asarray(-DURATION), DURATION))
        after = float(env(jnp.asarray(2 * DURATION), DURATION))
        assert before < 0.01
        assert after < 0.01

    def test_smooth_monotonic_ramp_up(self):
        env = SoftRectangularEnvelope(steepness=100.0)
        taus = jnp.linspace(0, DURATION / 2, 100)
        vals = jax.vmap(lambda tau: env(tau, DURATION))(taus)
        diffs = jnp.diff(vals)
        assert jnp.all(diffs >= -1e-10)

    def test_differentiable(self):
        env = SoftRectangularEnvelope(steepness=100.0)
        grad = jax.grad(lambda tau: env(tau, DURATION))(jnp.asarray(DURATION / 2))
        assert jnp.isfinite(grad)


class TestTukeyEnvelope:
    def test_zero_outside(self):
        env = TukeyEnvelope(alpha=0.5)
        assert float(env(jnp.asarray(-1e-6), DURATION)) == 0.0
        assert float(env(jnp.asarray(10e-6), DURATION)) == 0.0

    def test_flat_middle(self):
        env = TukeyEnvelope(alpha=0.2)
        # Middle should be exactly 1.0
        val = float(env(jnp.asarray(DURATION / 2), DURATION))
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_alpha_zero_is_rectangular(self):
        env = TukeyEnvelope(alpha=0.001)  # near-rectangular
        val = float(env(jnp.asarray(DURATION * 0.01), DURATION))
        assert val == pytest.approx(1.0, abs=0.1)

    def test_alpha_one_is_hann_like(self):
        tukey = TukeyEnvelope(alpha=1.0)
        hann = HannEnvelope()
        taus = jnp.linspace(0, DURATION, 100)
        tukey_vals = jax.vmap(lambda tau: tukey(tau, DURATION))(taus)
        hann_vals = jax.vmap(lambda tau: hann(tau, DURATION))(taus)
        assert jnp.allclose(tukey_vals, hann_vals, atol=1e-6)

    def test_symmetric(self):
        env = TukeyEnvelope(alpha=0.5)
        val_left = float(env(jnp.asarray(DURATION * 0.1), DURATION))
        val_right = float(env(jnp.asarray(DURATION * 0.9), DURATION))
        assert val_left == pytest.approx(val_right, rel=1e-6)
