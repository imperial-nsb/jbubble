"""Tests for jbubble.bubble.shell."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jbubble.bubble.property import ConstantProperty, Property
from jbubble.bubble.shell import (
    GompertzSurfaceTension,
    LipidShell,
    MarmottantSurfaceTension,
    NoShell,
    SmoothMarmottantSurfaceTension,
    ThickShell,
    _smooth_clamp_unit,
)
from jbubble.bubble.state import BubbleState

R0 = 2e-6
P_GAS0 = 173_325.0
SIGMA_R = 0.072
CHIS = np.geomspace(0.05, 2.0, 9)
RATIOS = [0.9, 0.95, 0.98, 0.995]


def _make_state(R_ratio, R_dot=0.0):
    return BubbleState(
        R=jnp.asarray(R_ratio * R0),
        R_dot=jnp.asarray(R_dot),
        R0=jnp.asarray(R0),
        P_gas0=jnp.asarray(P_GAS0),
    )


def _sigma_curve(law, ratios):
    """Evaluate `law` at R = ratios * R0."""
    R0_arr = jnp.asarray(R0)
    return jax.vmap(lambda x: law(BubbleState(R=x * R0, R0=R0_arr)))(
        jnp.asarray(ratios)
    )


def _dsigma_dx(law, ratios, R_b):
    """Return d sigma / d(R / R_b) at R = ratios * R0."""
    R0_arr = jnp.asarray(R0)
    f = jax.grad(lambda R: law(BubbleState(R=R, R0=R0_arr)))
    return jax.vmap(f)(jnp.asarray(ratios) * R0) * R_b


def _has_callback(fn, *args):
    """Return whether the jaxpr of `fn(*args)` contains a host callback."""
    return "callback" in str(jax.make_jaxpr(fn)(*args))


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
        st = MarmottantSurfaceTension(
            R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072
        )
        s = _make_state(0.95)  # well below buckle
        assert float(st(s)) == pytest.approx(0.0, abs=1e-10)

    def test_elastic_regime(self):
        st = MarmottantSurfaceTension(
            R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072
        )
        s = _make_state(1.0)  # R = R0, above R_buckle = 0.98*R0
        R_buckle = 0.98 * R0
        expected = 0.55 * ((R0 / R_buckle) ** 2 - 1.0)
        assert float(st(s)) == pytest.approx(expected, rel=1e-8)

    def test_ruptured_regime(self):
        st = MarmottantSurfaceTension(
            R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072
        )
        s = _make_state(1.5)  # large expansion → ruptured
        assert float(st(s)) == pytest.approx(0.072, rel=1e-8)

    def test_transitions_are_monotonic(self):
        st = MarmottantSurfaceTension(
            R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072
        )
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

    def test_chi_and_sigma_rupture_accept_a_property(self):
        plain = MarmottantSurfaceTension(
            R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072
        )
        wrapped = MarmottantSurfaceTension(
            R_buckle_ratio=0.98,
            chi=ConstantProperty(0.55),
            sigma_rupture=ConstantProperty(0.072),
        )
        assert isinstance(plain.chi, ConstantProperty)
        assert isinstance(plain.sigma_rupture, ConstantProperty)
        for ratio in (0.95, 1.0, 1.02, 1.5):
            s = _make_state(ratio)
            assert float(wrapped(s)) == float(plain(s))

    def test_gradient_with_respect_to_chi(self):
        def sigma(chi):
            st = MarmottantSurfaceTension(
                R_buckle_ratio=0.98, chi=chi, sigma_rupture=0.072
            )
            return st(_make_state(1.01))

        expected = (1.01 / 0.98) ** 2 - 1.0
        assert float(jax.grad(sigma)(0.55)) == pytest.approx(expected, rel=1e-12)

    def test_ratio_of_one_starts_buckled(self):
        st = MarmottantSurfaceTension(R_buckle_ratio=1.0, chi=0.55, sigma_rupture=0.072)
        assert float(st(_make_state(1.0))) == 0.0


class TestSmoothMarmottantSurfaceTension:
    @pytest.mark.parametrize("ratio", [0.9, 0.98, 1.0, 1.05])
    @pytest.mark.parametrize("chi", [0.05, 0.55, 2.0, 10.0])
    def test_bounds_and_monotonic(self, chi, ratio):
        # Includes initially buckled (ratio >= 1) and ruptured (chi = 10) bubbles.
        st = SmoothMarmottantSurfaceTension(
            R_buckle_ratio=ratio, chi=chi, sigma_rupture=SIGMA_R
        )
        s = _sigma_curve(st, np.linspace(0.01, 10.0, 20_001))
        assert bool(jnp.all(s >= 0.0)) and bool(jnp.all(s <= SIGMA_R))
        assert bool(jnp.all(jnp.diff(s) >= 0.0))

    @pytest.mark.parametrize("smoothing", [0.005, 0.01, 0.05])
    def test_max_error_is_chi_independent(self, smoothing):
        # sup |sigma - sigma_Marmottant| = smoothing * ln 2 * sigma_r for every
        # chi and ratio.
        bound = smoothing * math.log(2.0) * SIGMA_R
        for chi in CHIS:
            for ratio in RATIOS:
                kw = dict(R_buckle_ratio=ratio, chi=float(chi), sigma_rupture=SIGMA_R)
                x_b = ratio
                x_r = ratio * math.sqrt(1.0 + SIGMA_R / chi)
                # A dense grid that contains both corners exactly.
                x = np.unique(
                    np.concatenate([np.linspace(0.5, 3.0, 20_001), [x_b, x_r]])
                )
                err = jnp.abs(
                    _sigma_curve(
                        SmoothMarmottantSurfaceTension(**kw, smoothing=smoothing), x
                    )
                    - _sigma_curve(MarmottantSurfaceTension(**kw), x)
                )
                assert float(jnp.max(err)) <= bound * (1.0 + 1e-9)
                assert float(jnp.max(err)) >= 0.99 * bound

    def test_converges_to_marmottant(self):
        kw = dict(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=SIGMA_R)
        x = np.linspace(0.5, 3.0, 10_001)
        ref = _sigma_curve(MarmottantSurfaceTension(**kw), x)
        errs = [
            float(
                jnp.max(
                    jnp.abs(
                        _sigma_curve(
                            SmoothMarmottantSurfaceTension(**kw, smoothing=e), x
                        )
                        - ref
                    )
                )
            )
            for e in [1e-1, 1e-2, 1e-3, 1e-4]
        ]
        assert all(b < a for a, b in zip(errs, errs[1:], strict=False))
        assert errs[-1] < 1e-5

    def test_matches_elastic_law_away_from_corners(self):
        st = SmoothMarmottantSurfaceTension(
            R_buckle_ratio=0.98, chi=0.55, sigma_rupture=SIGMA_R
        )
        expected = 0.55 * ((1.0 / 0.98) ** 2 - 1.0)  # sigma_0 / sigma_r = 0.32
        assert float(_sigma_curve(st, [1.0])[0]) == pytest.approx(expected, rel=1e-9)

    def test_gradients_finite_for_all_inputs(self):
        def sigma(R, chi, ratio, s_r):
            st = SmoothMarmottantSurfaceTension(
                R_buckle_ratio=ratio, chi=chi, sigma_rupture=s_r
            )
            return st(BubbleState(R=R, R0=jnp.asarray(R0)))

        g = jax.vmap(
            jax.grad(sigma, argnums=(0, 1, 2, 3)), in_axes=(0, None, None, None)
        )
        Rs = jnp.asarray(np.linspace(0.01, 100.0, 5001) * R0)
        for chi, ratio in [(0.05, 0.9), (0.55, 0.98), (2.0, 0.995), (0.55, 1.02)]:
            for leaf in g(Rs, chi, ratio, SIGMA_R):
                assert bool(jnp.all(jnp.isfinite(leaf)))

    def test_jit_and_vmap_over_parameters(self):
        def sigma_at_R0(chi):
            st = SmoothMarmottantSurfaceTension(
                R_buckle_ratio=0.98, chi=chi, sigma_rupture=SIGMA_R
            )
            return st(BubbleState(R=jnp.asarray(1.01 * R0), R0=jnp.asarray(R0)))

        out = jax.jit(jax.vmap(sigma_at_R0))(jnp.asarray(CHIS))
        assert out.shape == (len(CHIS),) and bool(jnp.all(jnp.isfinite(out)))

    @pytest.mark.parametrize("smoothing", [0.01, 0.1, 0.5])
    def test_clamp_derivative_at_the_tie(self, smoothing):
        # f'(y) = sigmoid(y / eps) - sigmoid((y - 1) / eps), so f'(1/2) =
        # tanh(1 / (4 eps)). jnp.minimum(y, 1 - y) would give 0 here.
        grad = jax.grad(lambda y: _smooth_clamp_unit(y, smoothing))(0.5)
        assert float(grad) == pytest.approx(math.tanh(0.25 / smoothing), rel=1e-12)

    def test_clamp_derivative_is_continuous_across_the_tie(self):
        f_prime = jax.vmap(jax.grad(lambda y: _smooth_clamp_unit(y, 0.1)))
        y = jnp.asarray([0.5 - 1e-9, 0.5, 0.5 + 1e-9])
        assert np.ptp(np.asarray(f_prime(y))) < 1e-8

    @pytest.mark.parametrize("smoothing", [0.005, 0.01, 0.02])
    def test_bias_at_buckling_start(self, smoothing):
        # R_buckle_ratio = 1 starts the bubble at the buckling corner. The
        # smoothing shifts sigma(R0) to smoothing * ln 2 * sigma_r and halves
        # the elastic slope 2 chi / R0 of the piecewise law there. The
        # SmoothMarmottantSurfaceTension docstring documents this bias.
        chi = 0.55
        st = SmoothMarmottantSurfaceTension(
            R_buckle_ratio=1.0, chi=chi, sigma_rupture=SIGMA_R, smoothing=smoothing
        )
        state = _make_state(1.0)
        assert float(st(state)) == pytest.approx(
            smoothing * math.log(2.0) * SIGMA_R, rel=1e-12
        )
        slope = jax.grad(lambda R: st(BubbleState(R=R, R0=jnp.asarray(R0))))(
            jnp.asarray(R0)
        )
        assert float(slope) == pytest.approx(0.5 * 2.0 * chi / R0, rel=1e-12)

    @pytest.mark.parametrize("smoothing", [0.0, -0.01])
    def test_non_positive_smoothing_raises(self, smoothing):
        with pytest.raises(ValueError, match="smoothing > 0"):
            SmoothMarmottantSurfaceTension(
                R_buckle_ratio=0.98,
                chi=0.55,
                sigma_rupture=SIGMA_R,
                smoothing=smoothing,
            )

    def test_traced_smoothing_skips_the_check(self):
        def sigma(smoothing):
            st = SmoothMarmottantSurfaceTension(
                R_buckle_ratio=0.98,
                chi=0.55,
                sigma_rupture=SIGMA_R,
                smoothing=smoothing,
            )
            return st(_make_state(1.0))

        assert bool(jnp.isfinite(jax.jit(sigma)(0.01)))
        assert not _has_callback(sigma, 0.01)

    def test_chi_and_sigma_rupture_accept_a_state_dependent_property(self):
        class RadiusScaledChi(Property):
            # chi grows linearly with R / R0, a strain-stiffening shell.
            chi0: float

            def __call__(self, state):
                return self.chi0 * state.R / state.R0

        st = SmoothMarmottantSurfaceTension(
            R_buckle_ratio=0.98,
            chi=RadiusScaledChi(chi0=0.55),
            sigma_rupture=ConstantProperty(SIGMA_R),
        )
        x = 1.005
        expected = 0.55 * x * ((x / 0.98) ** 2 - 1.0)
        assert float(st(_make_state(x))) == pytest.approx(expected, rel=1e-9)

    def test_no_callback_in_jaxpr(self):
        def sigma(chi, sigma_r, ratio):
            st = SmoothMarmottantSurfaceTension(
                R_buckle_ratio=ratio, chi=chi, sigma_rupture=sigma_r
            )
            return st(_make_state(1.01))

        assert not _has_callback(sigma, 0.55, SIGMA_R, 0.98)


class TestSmoothMarmottantDynamics:
    def test_peak_expansion_matches_marmottant(self):
        from jbubble.bubble.eom import KellerMiksis
        from jbubble.bubble.gas import PolytropicGas
        from jbubble.bubble.medium import NewtonianMedium
        from jbubble.pulse.shapes import Sine
        from jbubble.pulse.tone_burst import ToneBurst
        from jbubble.simulation import run_simulation

        kw = dict(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=SIGMA_R)
        pulse = ToneBurst(freq=1e6, pressure=50e3, shape=Sine(), cycle_num=5)

        def peak(law):
            eom = KellerMiksis(
                gas=PolytropicGas(gamma=1.4),
                shell=LipidShell(sigma=law, kappa_s=2.4e-9),
                medium=NewtonianMedium(mu=1e-3),
                R0=R0,
                P_amb=101325.0,
                rho_L=998.0,
                c_L=1500.0,
            )
            return float(jnp.max(run_simulation(eom, pulse).radius)) / R0 - 1.0

        ref = peak(MarmottantSurfaceTension(**kw))
        smooth = peak(SmoothMarmottantSurfaceTension(**kw))
        assert smooth == pytest.approx(ref, rel=0.01)


class TestGompertzSurfaceTension:
    def test_well_posedness_raises_on_violation(self):
        # With R_buckle_ratio=0.85, sigma(R0) = chi*(1/0.85^2 - 1) = 0.55 * 0.384 = 0.211
        # which is > sigma_rupture=0.072
        with pytest.raises(ValueError, match="ill-posed"):
            GompertzSurfaceTension(R_buckle_ratio=0.85, chi=0.55, sigma_rupture=0.072)

    def test_well_posed_creates_successfully(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        assert st.R_buckle_ratio == 0.98
        assert isinstance(st.chi, ConstantProperty)
        assert isinstance(st.sigma_rupture, ConstantProperty)

    def test_sharpness_is_not_a_parameter(self):
        # The published law has no sharpness factor; jbubble 0.1's sharpness s
        # equals the published c scaled by s / e.
        with pytest.raises(TypeError):
            GompertzSurfaceTension(
                R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072, sharpness=1.0
            )

    def test_asymptotes_to_sigma_rupture(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        s = _make_state(3.0)  # large expansion
        assert float(st(s)) == pytest.approx(0.072, rel=1e-2)

    def test_smooth_monotonic(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
        values = _sigma_curve(st, np.linspace(0.5, 3.0, 2001))
        assert bool(jnp.all(jnp.diff(values) >= 0.0))
        assert bool(jnp.all(values >= 0.0)) and bool(jnp.all(values <= 0.072))

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


class TestGompertzSurfaceTensionPublished:
    """Checks against Gümmer, Schenke & Denner (2021), Eqs. 11 and 13-15."""

    @pytest.mark.parametrize("chi", [0.1, 0.5, 1.0])
    def test_max_slope_matches_marmottant_at_half_rupture(self, chi):
        # Eq. 15: the maximum Gompertz slope equals the Marmottant slope at
        # R_b sqrt(1 + sigma_r / (2 chi)).
        ratio = 0.995
        st = GompertzSurfaceTension(
            R_buckle_ratio=ratio, chi=chi, sigma_rupture=SIGMA_R
        )
        x = np.linspace(0.9, 2.0, 200_001)
        slope = _dsigma_dx(st, x, ratio * R0)
        expected = 2.0 * chi * math.sqrt(1.0 + SIGMA_R / (2.0 * chi))
        assert float(jnp.max(slope)) == pytest.approx(expected, rel=1e-6)

    def test_inflection_at_sigma_r_over_e(self):
        st = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=0.5, sigma_rupture=SIGMA_R)
        x = np.linspace(0.9, 1.3, 400_001)
        slope = _dsigma_dx(st, x, 0.98 * R0)
        sigma = _sigma_curve(st, x)
        assert float(sigma[jnp.argmax(slope)]) == pytest.approx(
            SIGMA_R / math.e, rel=1e-4
        )

    def test_paper_figure_1_parameters(self):
        # sigma_0 = 0.020, sigma_c = 0.072, chi = 0.5 (Gümmer et al. 2021, Fig. 1).
        ratio = 1.0 / math.sqrt(1.0 + 0.020 / 0.5)  # Eq. 11
        st = GompertzSurfaceTension(
            R_buckle_ratio=ratio, chi=0.5, sigma_rupture=SIGMA_R
        )
        assert float(_sigma_curve(st, [1.0])[0]) == pytest.approx(0.020, rel=1e-12)
        c = (2 * 0.5 * math.e / SIGMA_R) * math.sqrt(1 + SIGMA_R / (2 * 0.5))
        assert c == pytest.approx(39.09, abs=0.01)
        # Eqs. 13 and 14, written literally.
        R_b = ratio * R0
        b = -math.log(0.020 / SIGMA_R) / math.exp(c * (1.0 - R0 / R_b))
        for x in (0.95, 1.02, 1.2):
            expected = SIGMA_R * math.exp(-b * math.exp(c * (1.0 - x * R0 / R_b)))
            assert float(_sigma_curve(st, [x])[0]) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        "chi,ratio", [(0.05, 0.9), (0.55, 0.98), (1.7, 0.98), (2.0, 0.995)]
    )
    def test_sigma_at_R0_is_marmottant_value(self, chi, ratio):
        st = GompertzSurfaceTension(
            R_buckle_ratio=ratio, chi=chi, sigma_rupture=SIGMA_R
        )
        expected = chi * ((1.0 / ratio) ** 2 - 1.0)
        assert float(_sigma_curve(st, [1.0])[0]) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        "ratio,chi",
        [(0.85, 0.55), (1.0, 0.55), (1.02, 0.55), (0.0, 0.55), (-0.98, 0.55)],
    )
    def test_ill_posed_raises(self, ratio, chi):
        with pytest.raises(ValueError, match="strictly between"):
            GompertzSurfaceTension(R_buckle_ratio=ratio, chi=chi, sigma_rupture=SIGMA_R)

    def test_negative_chi_raises(self):
        # chi < 0 with R_buckle_ratio > 1 gives 0 < sigma_0 < sigma_r, but the
        # law is meaningless.
        with pytest.raises(ValueError, match="ill-posed"):
            GompertzSurfaceTension(R_buckle_ratio=1.02, chi=-0.5, sigma_rupture=SIGMA_R)

    def test_batched_parameters_are_checked(self):
        GompertzSurfaceTension(
            R_buckle_ratio=0.98, chi=jnp.asarray([0.3, 0.55]), sigma_rupture=SIGMA_R
        )
        with pytest.raises(ValueError, match="ill-posed"):
            GompertzSurfaceTension(
                R_buckle_ratio=jnp.asarray([0.98, 0.85]),
                chi=0.55,
                sigma_rupture=SIGMA_R,
            )

    def test_concrete_values_are_checked_inside_jit(self):
        # Python floats closed over by a jitted function are concrete, so the
        # check still runs, at trace time.
        def sigma(R):
            st = GompertzSurfaceTension(
                R_buckle_ratio=0.85, chi=0.55, sigma_rupture=SIGMA_R
            )
            return st(BubbleState(R=R, R0=jnp.asarray(R0)))

        with pytest.raises(ValueError, match="ill-posed"):
            jax.jit(sigma)(jnp.asarray(R0))

    def test_traced_values_skip_the_check_without_a_callback(self):
        def sigma(chi, sigma_r, ratio):
            st = GompertzSurfaceTension(
                R_buckle_ratio=ratio, chi=chi, sigma_rupture=sigma_r
            )
            return st(_make_state(1.01))

        expected = float(sigma(0.55, SIGMA_R, 0.98))
        assert float(jax.jit(sigma)(0.55, SIGMA_R, 0.98)) == pytest.approx(
            expected, rel=1e-12
        )
        assert not _has_callback(sigma, 0.55, SIGMA_R, 0.98)
        assert np.isfinite(float(jax.grad(sigma)(0.55, SIGMA_R, 0.98)))

    def test_state_dependent_property_skips_the_check(self):
        class Shifted(Property):
            def __call__(self, state):
                return 0.55 + 0.0 * state.R

        st = GompertzSurfaceTension(
            R_buckle_ratio=0.98, chi=Shifted(), sigma_rupture=SIGMA_R
        )
        ref = GompertzSurfaceTension(
            R_buckle_ratio=0.98, chi=0.55, sigma_rupture=SIGMA_R
        )
        assert float(st(_make_state(1.01))) == pytest.approx(
            float(ref(_make_state(1.01))), rel=1e-12
        )

    def test_gradients_finite_under_deep_compression(self):
        st = GompertzSurfaceTension(
            R_buckle_ratio=0.995, chi=2.0, sigma_rupture=SIGMA_R
        )
        x = np.linspace(0.01, 10.0, 2001)
        assert bool(jnp.all(jnp.isfinite(_dsigma_dx(st, x, 0.995 * R0))))
