"""Tests for jbubble.bubble.medium."""

import jax
import jax.numpy as jnp
import pytest

from jbubble.bubble.medium import (
    KelvinVoigtMedium,
    NeoHookeanMedium,
    NewtonianMedium,
    PowerLawMedium,
)
from jbubble.bubble.state import BubbleState

R0 = 2e-6


def _make_state(R_ratio, R_dot=0.0):
    return BubbleState(
        R=jnp.asarray(R_ratio * R0),
        R_dot=jnp.asarray(R_dot),
        R0=jnp.asarray(R0),
        P_gas0=jnp.asarray(1e5),
    )


class TestNewtonianMedium:
    def test_viscous_formula(self):
        mu = 1e-3
        medium = NewtonianMedium(mu=mu)
        R_dot = 0.5
        s = _make_state(1.0, R_dot=R_dot)
        expected = 4.0 * mu * R_dot / R0
        assert float(medium.p_viscous(s)) == pytest.approx(expected, rel=1e-10)

    def test_zero_elastic(self):
        medium = NewtonianMedium(mu=1e-3)
        s = _make_state(1.2, R_dot=0.5)
        assert float(medium.p_elastic(s)) == pytest.approx(0.0, abs=1e-15)

    def test_total_equals_viscous(self):
        medium = NewtonianMedium(mu=1e-3)
        s = _make_state(1.1, R_dot=0.3)
        assert float(medium(s)) == pytest.approx(float(medium.p_viscous(s)), rel=1e-10)

    def test_zero_velocity_zero_pressure(self):
        medium = NewtonianMedium(mu=1e-3)
        s = _make_state(1.0, R_dot=0.0)
        assert float(medium(s)) == pytest.approx(0.0, abs=1e-15)


class TestKelvinVoigtMedium:
    def test_viscous_formula(self):
        mu = 1e-3
        medium = KelvinVoigtMedium(mu=mu, G=1e3)
        R_dot = 0.5
        s = _make_state(1.0, R_dot=R_dot)
        expected = 4.0 * mu * R_dot / R0
        assert float(medium.p_viscous(s)) == pytest.approx(expected, rel=1e-10)

    def test_elastic_zero_at_equilibrium(self):
        medium = KelvinVoigtMedium(mu=1e-3, G=1e3)
        s = _make_state(1.0)
        assert float(medium.p_elastic(s)) == pytest.approx(0.0, abs=1e-6)

    def test_elastic_formula(self):
        G = 1e3
        medium = KelvinVoigtMedium(mu=1e-3, G=G)
        R_ratio = 1.2
        s = _make_state(R_ratio)
        expected = (4.0 / 3.0) * G * (R_ratio**3 - 1.0)
        assert float(medium.p_elastic(s)) == pytest.approx(expected, rel=1e-10)


class TestNeoHookeanMedium:
    def test_elastic_zero_at_equilibrium(self):
        medium = NeoHookeanMedium(mu=1e-3, G=1e3)
        s = _make_state(1.0)
        assert float(medium.p_elastic(s)) == pytest.approx(0.0, abs=1e-6)

    def test_elastic_formula(self):
        G = 1e3
        medium = NeoHookeanMedium(mu=1e-3, G=G)
        R_ratio = 1.3
        s = _make_state(R_ratio)
        beta = 1.0 / R_ratio
        expected = G * (2.5 - 2.0 * beta - 0.5 * beta**4)
        assert float(medium.p_elastic(s)) == pytest.approx(expected, rel=1e-10)

    def test_small_strain_matches_kv(self):
        """For small strains, NeoHookean should approximate KelvinVoigt."""
        G = 1e3
        neo = NeoHookeanMedium(mu=1e-3, G=G)
        kv = KelvinVoigtMedium(mu=1e-3, G=G)
        # Small strain: R = 1.01 * R0
        s = _make_state(1.01)
        neo_elastic = float(neo.p_elastic(s))
        kv_elastic = float(kv.p_elastic(s))
        assert neo_elastic == pytest.approx(kv_elastic, rel=0.05)

    def test_saturates_at_large_expansion(self):
        """Elastic pressure should approach 5G/2 for large R."""
        G = 1e3
        medium = NeoHookeanMedium(mu=1e-3, G=G)
        s = _make_state(1000.0)  # very large expansion
        assert float(medium.p_elastic(s)) == pytest.approx(2.5 * G, rel=1e-3)

    def test_strong_restoring_at_compression(self):
        """Elastic pressure should be strongly negative for R < R0."""
        medium = NeoHookeanMedium(mu=1e-3, G=1e3)
        s = _make_state(0.5)
        assert float(medium.p_elastic(s)) < 0


class TestPowerLawMedium:
    def test_n1_recovers_newtonian(self):
        """PowerLawMedium with n=1 should give the same result as NewtonianMedium."""
        mu = 1e-3
        power = PowerLawMedium(mu=mu, n_exp=1.0)
        newton = NewtonianMedium(mu=mu)
        s = _make_state(1.0, R_dot=0.5)
        assert float(power(s)) == pytest.approx(float(newton(s)), rel=1e-4)

    def test_shear_thinning_lower_viscous(self):
        """n < 1 (shear-thinning) gives lower viscous pressure at same shear rate."""
        s = _make_state(1.0, R_dot=0.5)
        newtonian = PowerLawMedium(mu=1e-3, n_exp=1.0)
        thinning = PowerLawMedium(mu=1e-3, n_exp=0.6)
        assert abs(float(thinning(s))) < abs(float(newtonian(s)))

    def test_zero_elastic(self):
        medium = PowerLawMedium(mu=1e-3, n_exp=0.8)
        s = _make_state(1.2, R_dot=0.3)
        assert float(medium.p_elastic(s)) == pytest.approx(0.0, abs=1e-15)

    def test_jit_compatible(self):
        medium = PowerLawMedium(mu=1e-3, n_exp=0.6)
        s = _make_state(1.0, R_dot=0.5)
        result = jax.jit(medium)(s)
        assert jnp.isfinite(result)

    def test_differentiable(self):
        medium = PowerLawMedium(mu=1e-3, n_exp=0.6)
        s = _make_state(1.0, R_dot=0.5)
        grad = jax.grad(medium)(s)
        assert jnp.isfinite(grad.R)
        assert jnp.isfinite(grad.R_dot)
