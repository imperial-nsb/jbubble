"""Tests for jbubble.bubble.gas."""

import jax
import jax.numpy as jnp
import pytest
from jbubble.bubble.gas import PolytropicGas, VanDerWaalsGas
from jbubble.bubble.state import BubbleState

P_GAS0 = 173_325.0  # typical equilibrium gas pressure
R0 = 2e-6


@pytest.fixture
def eq_state():
    return BubbleState(
        R=jnp.asarray(R0),
        R0=jnp.asarray(R0),
        P_gas0=jnp.asarray(P_GAS0),
    )


def _make_state(R, R_dot=0.0):
    return BubbleState(
        R=jnp.asarray(R),
        R_dot=jnp.asarray(R_dot),
        R0=jnp.asarray(R0),
        P_gas0=jnp.asarray(P_GAS0),
    )


class TestPolytropicGas:
    def test_equilibrium_returns_P_gas0(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        assert float(gas(eq_state)) == pytest.approx(P_GAS0, rel=1e-10)

    def test_expansion_lowers_pressure(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        expanded = _make_state(1.5 * R0)
        assert float(gas(expanded)) < P_GAS0

    def test_compression_raises_pressure(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        compressed = _make_state(0.8 * R0)
        assert float(gas(compressed)) > P_GAS0

    def test_isothermal_gamma_1(self, eq_state):
        gas = PolytropicGas(gamma=1.0)
        s = _make_state(1.5 * R0)
        expected = P_GAS0 * (R0 / (1.5 * R0)) ** 3
        assert float(gas(s)) == pytest.approx(expected, rel=1e-10)

    def test_adiabatic_gamma_1_4(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        R = 1.2 * R0
        s = _make_state(R)
        expected = P_GAS0 * (R0 / R) ** (3 * 1.4)
        assert float(gas(s)) == pytest.approx(expected, rel=1e-10)

    def test_jit_compatible(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        result = jax.jit(gas)(eq_state)
        assert float(result) == pytest.approx(P_GAS0, rel=1e-10)

    def test_grad_wrt_R(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        grad = jax.grad(gas)(eq_state)
        # dp/dR at equilibrium should be negative (expansion lowers pressure)
        assert float(grad.R) < 0


class TestVanDerWaalsGas:
    def test_equilibrium_returns_P_gas0(self, eq_state):
        gas = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        assert float(gas(eq_state)) == pytest.approx(P_GAS0, rel=1e-10)

    def test_zero_h_frac_recovers_polytropic(self):
        gas_vdw = VanDerWaalsGas(gamma=1.4, h_frac=0.0)
        gas_poly = PolytropicGas(gamma=1.4)
        s = _make_state(1.3 * R0)
        assert float(gas_vdw(s)) == pytest.approx(float(gas_poly(s)), rel=1e-8)

    def test_hard_core_raises_pressure_more(self):
        """VanDerWaals gives higher pressure than polytropic at same compression."""
        gas_vdw = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        gas_poly = PolytropicGas(gamma=1.4)
        s = _make_state(0.7 * R0)  # significant compression
        assert float(gas_vdw(s)) > float(gas_poly(s))

    def test_jit_compatible(self, eq_state):
        gas = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        result = jax.jit(gas)(eq_state)
        assert jnp.isfinite(result)

    def test_grad_wrt_R(self, eq_state):
        gas = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        grad = jax.grad(gas)(eq_state)
        assert float(grad.R) < 0


class TestIsAdmissible:
    @pytest.mark.parametrize(
        ("R", "expected"), [(R0, True), (0.0, False), (-R0, False)]
    )
    def test_polytropic_requires_a_positive_radius(self, R, expected):
        assert bool(PolytropicGas(gamma=1.4).is_admissible(_make_state(R))) is expected

    @pytest.mark.parametrize(
        ("ratio", "expected"), [(1.0, True), (0.21, True), (0.2, False), (0.1, False)]
    )
    def test_van_der_waals_excludes_the_hard_core(self, ratio, expected):
        gas = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        assert bool(gas.is_admissible(_make_state(ratio * R0))) is expected

    def test_nan_state_is_inadmissible(self):
        gas = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        assert not bool(gas.is_admissible(_make_state(jnp.nan)))
