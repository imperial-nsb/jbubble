"""Tests for jbubble.bubble.shell."""

import jax
import jax.numpy as jnp
import pytest

from jbubble.bubble.shell import (
    GompertzSurfaceTension,
    LipidShell,
    MarmottantSurfaceTension,
    NoShell,
    ThickShell,
)
from jbubble.bubble.state import BubbleState

R0 = 2e-6
P_GAS0 = 173_325.0


def _make_state(R_ratio, R_dot=0.0):
    return BubbleState(
        R=jnp.asarray(R_ratio * R0),
        R_dot=jnp.asarray(R_dot),
        R0=jnp.asarray(R0),
        P_gas0=jnp.asarray(P_GAS0),
    )


class TestNoShell:
    def test_laplace_pressure(self):
        shell = NoShell(sigma=0.072)
        s = _make_state(1.0)
        expected = 2.0 * 0.072 / R0
        assert float(shell(s)) == pytest.approx(expected, rel=1e-10)

    def test_zero_elastic(self):
        shell = NoShell(sigma=0.072)
        s = _make_state(1.0)
        assert float(shell.p_elastic(s)) == pytest.approx(0.0, abs=1e-15)

    def test_zero_viscous(self):
        shell = NoShell(sigma=0.072)
        s = _make_state(1.0, R_dot=1.0)
        assert float(shell.p_viscous(s)) == pytest.approx(0.0, abs=1e-15)

    def test_total_equals_laplace(self):
        shell = NoShell(sigma=0.072)
        s = _make_state(1.2, R_dot=0.5)
        assert float(shell(s)) == pytest.approx(float(shell.p_laplace(s)), rel=1e-10)


class TestLipidShell:
    def test_viscous_term(self):
        kappa_s = 2.4e-9
        shell = LipidShell(sigma=0.072, kappa_s=kappa_s)
        R_dot = 0.5
        s = _make_state(1.0, R_dot=R_dot)
        expected = 4.0 * kappa_s * R_dot / R0**2
        assert float(shell.p_viscous(s)) == pytest.approx(expected, rel=1e-10)

    def test_zero_elastic(self):
        shell = LipidShell(sigma=0.072, kappa_s=2.4e-9)
        s = _make_state(1.0)
        assert float(shell.p_elastic(s)) == pytest.approx(0.0, abs=1e-15)

    def test_total_is_laplace_plus_viscous(self):
        shell = LipidShell(sigma=0.072, kappa_s=2.4e-9)
        s = _make_state(1.1, R_dot=0.3)
        total = float(shell(s))
        laplace = float(shell.p_laplace(s))
        viscous = float(shell.p_viscous(s))
        assert total == pytest.approx(laplace + viscous, rel=1e-10)


class TestThickShell:
    def test_elastic_zero_at_equilibrium(self):
        shell = ThickShell(sigma=0.04, d_s=15e-9, G_s=10e6, mu_s=0.5)
        s = _make_state(1.0)
        assert float(shell.p_elastic(s)) == pytest.approx(0.0, abs=1e-6)

    def test_elastic_nonzero_away_from_equilibrium(self):
        shell = ThickShell(sigma=0.04, d_s=15e-9, G_s=10e6, mu_s=0.5)
        s = _make_state(1.2)
        assert float(shell.p_elastic(s)) != pytest.approx(0.0, abs=1e-3)

    def test_elastic_formula(self):
        d_s, G_s = 15e-9, 10e6
        shell = ThickShell(sigma=0.04, d_s=d_s, G_s=G_s, mu_s=0.5)
        R_ratio = 1.3
        s = _make_state(R_ratio)
        expected = (4.0 / 3.0) * G_s * (d_s / R0) * (1.0 - (1.0 / R_ratio) ** 3)
        assert float(shell.p_elastic(s)) == pytest.approx(expected, rel=1e-10)

    def test_viscous_formula(self):
        d_s, mu_s = 15e-9, 0.5
        shell = ThickShell(sigma=0.04, d_s=d_s, G_s=10e6, mu_s=mu_s)
        R_dot = 0.4
        R = 1.1 * R0
        s = _make_state(1.1, R_dot=R_dot)
        expected = 4.0 * mu_s * d_s * R_dot / R**2
        assert float(shell.p_viscous(s)) == pytest.approx(expected, rel=1e-10)


class TestMarmottantSurfaceTension:
    def test_buckled_regime(self):
        st = MarmottantSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        s = _make_state(0.95)  # well below buckle
        assert float(st(s)) == pytest.approx(0.0, abs=1e-10)

    def test_elastic_regime(self):
        st = MarmottantSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        s = _make_state(1.0)  # R = R0, above R_buckle = 0.98*R0
        R_buckle = 0.98 * R0
        expected = 0.55 * ((R0 / R_buckle) ** 2 - 1.0)
        assert float(st(s)) == pytest.approx(expected, rel=1e-8)

    def test_ruptured_regime(self):
        st = MarmottantSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        s = _make_state(1.5)  # large expansion → ruptured
        assert float(st(s)) == pytest.approx(0.072, rel=1e-8)

    def test_transitions_are_monotonic(self):
        st = MarmottantSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        R0_arr = jnp.asarray(R0)
        P_gas0_arr = jnp.asarray(P_GAS0)
        ratios = jnp.linspace(0.9, 1.5, 100)
        Rs = ratios * R0

        def eval_sigma(R_val):
            s = BubbleState(R=R_val, R0=R0_arr, P_gas0=P_gas0_arr)
            return st(s)

        values = jax.vmap(eval_sigma)(Rs)
        # Surface tension should be non-decreasing (buckled→elastic→ruptured)
        diffs = jnp.diff(values)
        assert jnp.all(diffs >= -1e-10)


class TestGompertzSurfaceTension:
    def test_well_posedness_raises_on_violation(self):
        # With R_buckle_ratio=0.85, sigma(R0) = chi*(1/0.85^2 - 1) = 0.55 * 0.384 = 0.211
        # which is > sigma_rupture=0.072
        with pytest.raises(ValueError, match="ill-posed"):
            GompertzSurfaceTension(
                R_buckle_ratio=0.85, chi=0.55, sigma_rupture=0.072
            )

    def test_well_posed_creates_successfully(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        assert st.R_buckle_ratio == 0.98

    def test_sigma_at_R0_matches_elastic_regime(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        s = _make_state(1.0)
        # Should approximate chi * ((1/0.98)^2 - 1)
        expected = 0.55 * ((1.0 / 0.98) ** 2 - 1.0)
        assert float(st(s)) == pytest.approx(expected, rel=1e-4)

    def test_asymptotes_to_sigma_rupture(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        s = _make_state(3.0)  # large expansion
        assert float(st(s)) == pytest.approx(0.072, rel=1e-2)

    def test_smooth_monotonic(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        R0_arr = jnp.asarray(R0)
        P_gas0_arr = jnp.asarray(P_GAS0)
        ratios = jnp.linspace(0.95, 1.5, 200)
        Rs = ratios * R0

        def eval_sigma(R_val):
            s = BubbleState(R=R_val, R0=R0_arr, P_gas0=P_gas0_arr)
            return st(s)

        values = jax.vmap(eval_sigma)(Rs)
        diffs = jnp.diff(values)
        # Should be monotonically increasing in the elastic→ruptured transition
        assert jnp.all(diffs >= -1e-10)

    def test_differentiable(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        s = _make_state(1.0)
        grad = jax.grad(st)(s)
        assert jnp.isfinite(grad.R)
        assert float(grad.R) > 0  # surface tension increases with R in elastic regime

    def test_jit_compatible(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        s = _make_state(1.0)
        result = jax.jit(st)(s)
        assert jnp.isfinite(result)
