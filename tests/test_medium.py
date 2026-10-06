"""Tests for jbubble.bubble.medium.

The elastic and power-law tests check each model against an independent
quadrature of its constitutive law, rather than restating the closed form.
For incompressible radial motion, the medium's contribution to the wall
pressure is

    p_medium = -2 * integral_R^inf (tau_rr - tau_tt)(r) / r dr,

where tau is the deviatoric stress (Yang & Church 2005; Warnez & Johnsen 2015).
"""

import jax
import jax.numpy as jnp
import numpy as np
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


def _wall_pressure_quadrature(R, stress_difference, nodes=400):
    """Return -2 * integral_R^inf stress_difference(s) / r dr.

    ``stress_difference`` takes ``s = R / r`` in (0, 1].  With ``s = w**4``,
    ``dr / r = -ds / s = -4 dw / w``, which keeps the integrand smooth at
    ``r -> inf`` for every model tested here.  Gauss-Legendre on (0, 1).
    """
    x, weights = np.polynomial.legendre.leggauss(nodes)
    w = 0.5 * (x + 1.0)
    s = w**4
    return -2.0 * np.sum(0.5 * weights * stress_difference(s) * 4.0 / w)


# ── Newtonian ────────────────────────────────────────────────────────────────


class TestNewtonianMedium:
    def test_viscous_matches_quadrature(self):
        """Newtonian stress difference: 2 mu (D_rr - D_tt) = -6 mu R^2 Rdot / r^3."""
        mu, R_dot, R = 1e-3, 0.5, 1.3 * R0
        expected = _wall_pressure_quadrature(R, lambda s: -6.0 * mu * R_dot / R * s**3)
        medium = NewtonianMedium(mu=mu)
        s = _make_state(1.3, R_dot=R_dot)
        assert float(medium.p_viscous(s)) == pytest.approx(expected, rel=1e-12)

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


# ── Kelvin-Voigt (Yang & Church 2005) ────────────────────────────────────────

_RATIOS = [0.3, 0.5, 0.8, 0.95, 1.05, 1.2, 1.5, 2.0, 3.0, 5.0]


def _linear_elastic_stress_difference(R, G):
    """Small-strain Hookean solid with the displacement u = (R^3 - R0^3)/(3 r^2).

    eps_rr = du/dr = -2(R^3 - R0^3)/(3 r^3) and eps_tt = u/r, so
    tau_rr - tau_tt = 2 G (eps_rr - eps_tt) = -2 G (R^3 - R0^3) / r^3.
    """

    def stress_difference(s):
        u_over_r = (R**3 - R0**3) / (3.0 * R**3) * s**3
        eps_rr = -2.0 * u_over_r
        eps_tt = u_over_r
        return 2.0 * G * (eps_rr - eps_tt)

    return stress_difference


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
        assert float(medium.p_elastic(s)) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("ratio", _RATIOS)
    def test_elastic_matches_linear_elastic_quadrature(self, ratio):
        G = 1e5
        R = ratio * R0
        expected = _wall_pressure_quadrature(R, _linear_elastic_stress_difference(R, G))
        medium = KelvinVoigtMedium(mu=0.0, G=G)
        assert float(medium.p_elastic(_make_state(ratio))) == pytest.approx(
            expected, rel=1e-10
        )

    def test_elastic_bounded_under_expansion(self):
        """The Yang-Church elastic pressure saturates at 4G/3 as R grows."""
        G = 1e5
        medium = KelvinVoigtMedium(mu=0.0, G=G)
        assert float(medium.p_elastic(_make_state(100.0))) == pytest.approx(
            4.0 * G / 3.0, rel=1e-5
        )

    def test_small_strain_slope_matches_neo_hookean(self):
        """Both media linearise to 4 G (R - R0) / R0."""
        G = 1e5

        def slope(medium):
            return jax.grad(
                lambda R: medium.p_elastic(BubbleState(R=R, R0=jnp.asarray(R0)))
            )(jnp.asarray(R0))

        kv = slope(KelvinVoigtMedium(mu=0.0, G=G))
        nh = slope(NeoHookeanMedium(mu=0.0, G=G))
        assert float(kv) == pytest.approx(4.0 * G / R0, rel=1e-12)
        assert float(nh) == pytest.approx(4.0 * G / R0, rel=1e-12)


# ── Neo-Hookean ──────────────────────────────────────────────────────────────


def _neo_hookean_stress_difference(R, G):
    """Incompressible neo-Hookean: tau_rr - tau_tt = G (lam^-4 - lam^2).

    lam = r / r0 is the hoop stretch of the material point now at r, with
    r^3 - r0^3 = R^3 - R0^3, so lam^3 = 1 / (1 - (R^3 - R0^3) s^3 / R^3).
    """

    def stress_difference(s):
        lam = (1.0 - (R**3 - R0**3) / R**3 * s**3) ** (-1.0 / 3.0)
        return G * (lam**-4 - lam**2)

    return stress_difference


class TestNeoHookeanMedium:
    def test_elastic_zero_at_equilibrium(self):
        medium = NeoHookeanMedium(mu=1e-3, G=1e3)
        s = _make_state(1.0)
        assert float(medium.p_elastic(s)) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("ratio", _RATIOS)
    def test_elastic_matches_finite_strain_quadrature(self, ratio):
        G = 1e5
        R = ratio * R0
        expected = _wall_pressure_quadrature(R, _neo_hookean_stress_difference(R, G))
        medium = NeoHookeanMedium(mu=0.0, G=G)
        assert float(medium.p_elastic(_make_state(ratio))) == pytest.approx(
            expected, rel=1e-9
        )

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


# ── Power law ────────────────────────────────────────────────────────────────


def _power_law_stress_difference(R, R_dot, K, n):
    """eta = K gdot^(n-1) with gdot = sqrt(2 D:D) for u = R^2 Rdot / r^2.

    D = diag(-2a, a, a) with a = R^2 Rdot / r^3, so gdot = 2 sqrt(3) |a| and
    tau_rr - tau_tt = 2 eta (D_rr - D_tt) = -6 eta a.
    """

    def stress_difference(s):
        a = R_dot / R * s**3
        D_rr, D_tt = -2.0 * a, a
        gdot = np.sqrt(2.0 * (D_rr**2 + 2.0 * D_tt**2))
        eta = K * gdot ** (n - 1.0)
        return 2.0 * eta * (D_rr - D_tt)

    return stress_difference


class TestPowerLawMedium:
    @pytest.mark.parametrize("n", [0.4, 0.6, 0.8, 1.0, 1.3])
    @pytest.mark.parametrize("R_dot", [5.0, -2.0])
    def test_viscous_matches_rheometric_quadrature(self, n, R_dot):
        """With the default eps, the wall shear rate here is ~5e6 s^-1 >> eps."""
        K, ratio = 0.017, 1.0
        R = ratio * R0
        expected = _wall_pressure_quadrature(
            R, _power_law_stress_difference(R, R_dot, K, n)
        )
        medium = PowerLawMedium(mu=K, n_exp=n)
        got = float(medium.p_viscous(_make_state(ratio, R_dot=R_dot)))
        assert got == pytest.approx(expected, rel=1e-8)

    def test_default_eps(self):
        assert PowerLawMedium(mu=1e-3, n_exp=0.7).eps == 1e2

    def test_n1_recovers_newtonian_exactly(self):
        mu = 1e-3
        power = PowerLawMedium(mu=mu, n_exp=1.0)
        newton = NewtonianMedium(mu=mu)
        for R_dot in [0.0, 1e-6, 0.5, -3.0]:
            s = _make_state(1.1, R_dot=R_dot)
            assert float(power(s)) == pytest.approx(float(newton(s)), rel=1e-14)

    def test_shear_thinning_lower_viscous(self):
        """n < 1 (shear-thinning) gives lower viscous pressure at same shear rate."""
        s = _make_state(1.0, R_dot=0.5)
        newtonian = PowerLawMedium(mu=1e-3, n_exp=1.0)
        thinning = PowerLawMedium(mu=1e-3, n_exp=0.6)
        assert abs(float(thinning(s))) < abs(float(newtonian(s)))

    def test_regularised_through_zero_velocity(self):
        """At Rdot = 0 the pressure is zero and its slope finite: K eps^(n-1) 4/(n R)."""
        K, n, eps = 0.017, 0.5, 1e2
        medium = PowerLawMedium(mu=K, n_exp=n, eps=eps)

        def p_of(R_dot):
            state = BubbleState(R=jnp.asarray(R0), R_dot=R_dot, R0=jnp.asarray(R0))
            return medium.p_viscous(state)

        assert float(p_of(jnp.asarray(0.0))) == 0.0
        slope = float(jax.grad(p_of)(jnp.asarray(0.0)))
        assert slope == pytest.approx(4.0 * K / n * eps ** (n - 1.0) / R0, rel=1e-12)

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


class TestLiquidProperties:
    """`rho_L` and `c_L` default to water at 20 °C and stay keyword-only."""

    @pytest.mark.parametrize(
        "medium",
        [
            NewtonianMedium(mu=1e-3),
            KelvinVoigtMedium(mu=1e-3, G=1e3),
            NeoHookeanMedium(mu=1e-3, G=1e3),
            PowerLawMedium(mu=1e-3, n_exp=0.7),
        ],
        ids=lambda m: type(m).__name__,
    )
    def test_defaults_are_water(self, medium):
        # NIST WebBook: 998.2 kg/m³ at 20 °C; 1500 m/s is the round value
        # that ultrasound modelling uses.
        assert medium.rho_L == 998.0
        assert medium.c_L == 1500.0

    def test_keyword_only(self):
        with pytest.raises(TypeError):
            NewtonianMedium(1e-3, 1000.0)  # ty: ignore[too-many-positional-arguments]
        medium = KelvinVoigtMedium(mu=1e-3, G=1e3, rho_L=1060.0, c_L=1540.0)
        assert (medium.rho_L, medium.c_L) == (1060.0, 1540.0)
