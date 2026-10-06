"""Tests for jbubble.bubble.eom."""

import equinox as eqx
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


def _common_args(c_L=C_L):
    return dict(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=SIGMA),
        medium=NewtonianMedium(mu=MU, rho_L=RHO_L, c_L=c_L),
        R0=R0,
        P_amb=P_AMB,
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
        eom = KellerMiksis(**_common_args())
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
        eom = ModifiedRayleighPlesset(**_common_args())
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert isinstance(result, BubbleState)

    def test_equilibrium_nearly_zero_accel(self):
        eom = ModifiedRayleighPlesset(**_common_args())
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert float(result.R_dot) == pytest.approx(0.0, abs=1e-2)


class TestKellerMiksis:
    def test_returns_bubble_state(self):
        eom = KellerMiksis(**_common_args())
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert isinstance(result, BubbleState)

    def test_equilibrium_nearly_zero_accel(self):
        eom = KellerMiksis(**_common_args())
        s = eom.initial_state()
        result = eom(jnp.asarray(0.0), s, _zero_pulse)
        assert float(result.R_dot) == pytest.approx(0.0, abs=1e-2)

    def test_large_c_L_approaches_RP(self):
        """As c_L → ∞, KellerMiksis should approach RayleighPlesset."""
        rp = RayleighPlesset(**_common_args())
        km = KellerMiksis(**_common_args(c_L=1e10))  # effectively infinite c_L

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
        eom = KellerMiksis(**_common_args())
        s = eom.initial_state()

        def loss(s):
            return eom(jnp.asarray(0.0), s, _zero_pulse).R_dot

        grad = jax.grad(loss)(s)
        assert jnp.isfinite(grad.R)


class TestMediumLiquid:
    """The equations of motion read rho_L and c_L from the medium."""

    _moving = BubbleState(
        R=jnp.asarray(1.2 * R0),
        R_dot=jnp.asarray(0.5),
        R0=jnp.asarray(R0),
        P_gas0=jnp.asarray(P_AMB + 2 * SIGMA / R0),
    )

    @pytest.mark.parametrize(
        "cls", [RayleighPlesset, ModifiedRayleighPlesset, KellerMiksis]
    )
    def test_no_liquid_fields_on_the_eom(self, cls):
        eom = cls(**_common_args())
        assert not hasattr(eom, "rho_L")
        assert not hasattr(eom, "c_L")

    def test_rayleigh_plesset_uses_medium_density(self):
        """At rest, R_ddot = (p_L - P_amb) / (rho_L R) scales as 1 / rho_L."""
        state = BubbleState(
            R=jnp.asarray(1.2 * R0),
            R0=jnp.asarray(R0),
            P_gas0=jnp.asarray(P_AMB + 2 * SIGMA / R0),
        )
        eom = RayleighPlesset(**_common_args())
        dense = eqx.tree_at(lambda e: e.medium.rho_L, eom, 2 * RHO_L)
        a = eom(jnp.asarray(0.0), state, _zero_pulse).R_dot
        b = dense(jnp.asarray(0.0), state, _zero_pulse).R_dot
        assert float(b) == pytest.approx(float(a) / 2, rel=1e-12)

    @pytest.mark.parametrize("cls", [ModifiedRayleighPlesset, KellerMiksis])
    def test_sound_speed_comes_from_medium(self, cls):
        eom = cls(**_common_args())
        faster = eqx.tree_at(lambda e: e.medium.c_L, eom, 3000.0)
        built = cls(**_common_args(c_L=3000.0))
        a = faster(jnp.asarray(0.0), self._moving, _zero_pulse).R_dot
        b = built(jnp.asarray(0.0), self._moving, _zero_pulse).R_dot
        c = eom(jnp.asarray(0.0), self._moving, _zero_pulse).R_dot
        assert float(a) == float(b)
        assert float(a) != float(c)

    def test_state_scale_uses_medium_density(self):
        eom = KellerMiksis(**_common_args())
        dense = eqx.tree_at(lambda e: e.medium.rho_L, eom, 4 * RHO_L)
        v = eom.state_scale(self._moving).R_dot
        v_dense = dense.state_scale(self._moving).R_dot
        assert float(v) == pytest.approx((P_AMB / RHO_L) ** 0.5, rel=1e-12)
        assert float(v_dense) == pytest.approx(float(v) / 2, rel=1e-12)

    def test_tree_at_under_jit_and_grad(self):
        eom = KellerMiksis(**_common_args())

        @jax.jit
        def accel(rho_L, c_L):
            e = eqx.tree_at(lambda e: e.medium.rho_L, eom, rho_L)
            e = eqx.tree_at(lambda e: e.medium.c_L, e, c_L)
            return e(jnp.asarray(0.0), self._moving, _zero_pulse).R_dot

        rho, c = jnp.asarray(RHO_L), jnp.asarray(C_L)
        eager = eom(jnp.asarray(0.0), self._moving, _zero_pulse).R_dot
        assert float(accel(rho, c)) == pytest.approx(float(eager), rel=1e-12)
        d_rho, d_c = jax.grad(accel, argnums=(0, 1))(rho, c)
        h_rho, h_c = 1e-3 * RHO_L, 1e-3 * C_L
        fd_rho = (accel(rho + h_rho, c) - accel(rho - h_rho, c)) / (2 * h_rho)
        fd_c = (accel(rho, c + h_c) - accel(rho, c - h_c)) / (2 * h_c)
        assert float(d_rho) != 0.0 and float(d_c) != 0.0
        assert float(d_rho) == pytest.approx(float(fd_rho), rel=1e-5)
        assert float(d_c) == pytest.approx(float(fd_c), rel=1e-5)
