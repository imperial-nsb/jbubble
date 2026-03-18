"""Tests for jbubble.metrics."""

import jax
import jax.numpy as jnp
import pytest
from jbubble.metrics import (
    mse_emission,
    mse_radius,
    normalised_mse_emission,
    normalised_mse_radius,
    peak_expansion,
    peak_expansion_error,
)


class TestMseRadius:
    def test_zero_for_identical(self):
        r = jnp.array([1.0, 2.0, 3.0])
        assert float(mse_radius(r, r)) == pytest.approx(0.0, abs=1e-15)

    def test_known_value(self):
        r_sim = jnp.array([1.0, 2.0, 3.0])
        r_target = jnp.array([2.0, 3.0, 4.0])
        # MSE = mean((1)^2 + (1)^2 + (1)^2) = 1.0
        assert float(mse_radius(r_sim, r_target)) == pytest.approx(1.0, abs=1e-10)

    def test_differentiable(self):
        r_sim = jnp.array([1.0, 2.0])
        r_target = jnp.array([1.5, 2.5])
        grad = jax.grad(lambda r: mse_radius(r, r_target).sum())(r_sim)
        assert jnp.all(jnp.isfinite(grad))


class TestNormalisedMseRadius:
    def test_zero_for_identical(self):
        r = jnp.array([1.0, 2.0, 3.0])
        assert float(normalised_mse_radius(r, r, 1.0)) == pytest.approx(0.0, abs=1e-15)

    def test_equivalent_to_scaled_mse(self):
        r_sim = jnp.array([1e-6, 2e-6, 3e-6])
        r_target = jnp.array([1.1e-6, 2.1e-6, 3.1e-6])
        R0 = 2e-6
        norm = float(normalised_mse_radius(r_sim, r_target, R0))
        raw = float(mse_radius(r_sim / R0, r_target / R0))
        assert norm == pytest.approx(raw, rel=1e-10)


class TestPeakExpansion:
    def test_known_value(self):
        r_sim = jnp.array([2e-6, 4e-6, 3e-6, 1e-6])
        R0 = 2e-6
        assert float(peak_expansion(r_sim, R0)) == pytest.approx(2.0, rel=1e-10)


class TestPeakExpansionError:
    def test_zero_when_matched(self):
        r_sim = jnp.array([2e-6, 4e-6, 3e-6])
        R0 = 2e-6
        assert float(peak_expansion_error(r_sim, R0, 2.0)) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_nonzero_when_mismatched(self):
        r_sim = jnp.array([2e-6, 4e-6, 3e-6])
        R0 = 2e-6
        err = float(peak_expansion_error(r_sim, R0, 3.0))
        assert err == pytest.approx(1.0, rel=1e-10)  # (2-3)^2 = 1

    def test_differentiable(self):
        r_sim = jnp.array([2e-6, 4e-6, 3e-6])
        R0 = 2e-6
        grad = jax.grad(lambda r: peak_expansion_error(r, R0, 2.5))(r_sim)
        assert jnp.all(jnp.isfinite(grad))


class TestMseEmission:
    def test_zero_for_identical(self):
        p = jnp.array([100.0, 200.0, 300.0])
        assert float(mse_emission(p, p)) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        p_sim = jnp.array([100.0, 200.0])
        p_target = jnp.array([110.0, 220.0])
        expected = float(jnp.mean(jnp.array([10.0**2, 20.0**2])))
        assert float(mse_emission(p_sim, p_target)) == pytest.approx(
            expected, rel=1e-10
        )


class TestNormalisedMseEmission:
    def test_zero_for_identical(self):
        p = jnp.array([100.0, 200.0])
        assert float(normalised_mse_emission(p, p, 100.0)) == pytest.approx(
            0.0, abs=1e-15
        )

    def test_scaling(self):
        p_sim = jnp.array([100.0, 200.0])
        p_target = jnp.array([110.0, 220.0])
        p_ref = 100.0
        raw = float(mse_emission(p_sim / p_ref, p_target / p_ref))
        norm = float(normalised_mse_emission(p_sim, p_target, p_ref))
        assert norm == pytest.approx(raw, rel=1e-10)
