"""Tests for jbubble.bubble.eom."""

import jax
import jax.numpy as jnp
import pytest
from jbubble.bubble.eom import (
    KellerMiksis,
    ModifiedRayleighPlesset,
    RayleighPlesset,
)
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.bubble.state import BubbleState

R0 = 2e-6
P_AMB = 101325.0
RHO_L = 998.0
C_L = 1500.0
SIGMA = 0.072
MU = 1e-3


def _zero_pulse(t):
    return t * 0.0


def _common_args():
    return dict(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=SIGMA),
        medium=NewtonianMedium(mu=MU),
        R0=R0,
        P_amb=P_AMB,
        rho_L=RHO_L,
    )


class TestInitialState:
    def test_R_equals_R0(self):
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state()
        assert float(s.R) == pytest.approx(R0, rel=1e-10)

    def test_R_dot_is_zero(self):
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state()
        assert float(s.R_dot) == pytest.approx(0.0, abs=1e-15)

    def test_R0_in_state(self):
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state()
        assert float(s.R0) == pytest.approx(R0, rel=1e-10)

    def test_P_gas0_includes_laplace(self):
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state()
        expected = P_AMB + 2.0 * SIGMA / R0
        assert float(s.P_gas0) == pytest.approx(expected, rel=1e-8)

    def test_keywords_start_away_from_equilibrium(self):
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state(R=1.2 * R0, R_dot=-0.5)
        equilibrium = eom.initial_state()
        assert float(s.R) == 1.2 * R0
        assert float(s.R_dot) == -0.5
        assert float(s.R0) == float(equilibrium.R0)
        assert float(s.P_gas0) == float(equilibrium.P_gas0)

    def test_keywords_are_traceable(self):
        eom = RayleighPlesset(**_common_args())
        R = jax.jit(lambda r: eom.initial_state(R=r).R)(jnp.asarray(3e-6))
        assert float(R) == 3e-6


class TestIsAdmissible:
    @pytest.mark.parametrize(
        ("R", "expected"), [(R0, True), (1e-12, True), (0.0, False), (-R0, False)]
    )
    def test_requires_a_positive_radius(self, R, expected):
        eom = KellerMiksis(**_common_args(), c_L=C_L)
        s = eom.initial_state(R=R)
        assert bool(eom.is_admissible(s)) is expected

    def test_defers_to_the_gas(self):
        from jbubble.bubble.gas import VanDerWaalsGas

        args = _common_args() | {"gas": VanDerWaalsGas(gamma=1.4, h_frac=0.25)}
        eom = RayleighPlesset(**args)
        assert not bool(eom.is_admissible(eom.initial_state(R=0.2 * R0)))
        assert bool(eom.is_admissible(eom.initial_state(R=0.3 * R0)))


class TestPL:
    def test_p_L_at_equilibrium(self):
        """At equilibrium with zero velocity, p_L should equal P_amb + laplace."""
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state()
        p_L = float(eom.p_L(s))
        # gas pressure = P_gas0 = P_amb + 2σ/R0
        # shell pressure = 2σ/R0 (NoShell = Laplace only)
        # medium = 0 (zero velocity)
        # p_L = gas - shell - medium = (P_amb + 2σ/R0) - 2σ/R0 - 0 = P_amb
        assert p_L == pytest.approx(P_AMB, rel=1e-8)


class TestRayleighPlesset:
    def test_returns_bubble_state(self):
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert isinstance(result, BubbleState)

    def test_R_derivative_is_R_dot(self):
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert float(result.R) == pytest.approx(float(s.R_dot), abs=1e-15)

    def test_equilibrium_zero_driving_nearly_zero_accel(self):
        """At equilibrium with no driving, R̈ should be approximately zero."""
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert float(result.R_dot) == pytest.approx(0.0, abs=1e-2)

    def test_jit_compatible(self):
        eom = RayleighPlesset(**_common_args())
        s = eom.initial_state()
        result = jax.jit(lambda: eom(jnp.asarray(0.0), s, _zero_pulse))()
        assert jnp.isfinite(result.R_dot)


class TestModifiedRayleighPlesset:
    def test_returns_bubble_state(self):
        eom = ModifiedRayleighPlesset(**_common_args(), c_L=C_L)
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert isinstance(result, BubbleState)

    def test_equilibrium_nearly_zero_accel(self):
        eom = ModifiedRayleighPlesset(**_common_args(), c_L=C_L)
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert float(result.R_dot) == pytest.approx(0.0, abs=1e-2)


class TestKellerMiksis:
    def test_returns_bubble_state(self):
        eom = KellerMiksis(**_common_args(), c_L=C_L)
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert isinstance(result, BubbleState)

    def test_equilibrium_nearly_zero_accel(self):
        eom = KellerMiksis(**_common_args(), c_L=C_L)
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert float(result.R_dot) == pytest.approx(0.0, abs=1e-2)

    def test_large_c_L_approaches_RP(self):
        """As c_L → ∞, KellerMiksis should approach RayleighPlesset."""
        args = _common_args()
        rp = RayleighPlesset(**args)
        km = KellerMiksis(**args, c_L=1e10)  # effectively infinite c_L

        s = BubbleState(
            R=jnp.asarray(1.5 * R0),
            R_dot=jnp.asarray(0.1),
            R0=jnp.asarray(R0),
            P_gas0=jnp.asarray(P_AMB + 2.0 * SIGMA / R0),
        )
        rp_result = rp(jnp.asarray(0.0), s, _zero_pulse)
        km_result = km(jnp.asarray(0.0), s, _zero_pulse)
        assert float(km_result.R_dot) == pytest.approx(float(rp_result.R_dot), rel=1e-4)

    def test_differentiable(self):
        eom = KellerMiksis(**_common_args(), c_L=C_L)
        s = eom.initial_state()

        def loss(s):
            return eom(jnp.asarray(0.0), s, _zero_pulse).R_dot

        grad = jax.grad(loss)(s)
        assert jnp.isfinite(grad.R)
