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


class RadiusScaledChi(Property):
    """A chi that grows linearly with R / R0, as for a strain-stiffening shell."""

    chi0: float

    def __call__(self, state):
        return self.chi0 * state.R / state.R0


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


def _shell_volume(d_s, R0_=R0):
    """Return V_s = R0^3 - (R0 - d_s)^3, computed directly."""
    return R0_**3 - (R0_ - d_s) ** 3


def _linear_coefficients(eom):
    """Return (omega0^2, beta) of the EoM linearised about equilibrium.

    The undriven ODE linearises to x'' + 2 beta x' + omega0^2 x = 0, so the
    Jacobian of (R, R_dot) -> (R_dot, R_ddot) has -omega0^2 and -2 beta in its
    second row.
    """
    y0 = eom.initial_state()

    def rhs(v):
        st = BubbleState(R=v[0], R_dot=v[1], R0=y0.R0, P_gas0=y0.P_gas0)
        d = eom(jnp.asarray(0.0), st, lambda t: 0.0 * t)
        return jnp.stack([d.R, d.R_dot])

    J = jax.jacfwd(rhs)(jnp.stack([y0.R, y0.R_dot]))
    return float(-J[1, 0]), float(-J[1, 1] / 2.0)


class TestThickShell:
    G_S, MU_S, SIGMA = 11.7e6, 0.45, 0.04
    GAMMA, P_AMB, RHO, MU = 1.4, 101325.0, 998.0, 1e-3

    def _eom(self, d_s):
        from jbubble.bubble.eom import RayleighPlesset
        from jbubble.bubble.gas import PolytropicGas
        from jbubble.bubble.medium import NewtonianMedium

        return RayleighPlesset(
            gas=PolytropicGas(gamma=self.GAMMA),
            shell=ThickShell(sigma=self.SIGMA, d_s=d_s, G_s=self.G_S, mu_s=self.MU_S),
            medium=NewtonianMedium(mu=self.MU, rho_L=self.RHO),
            R0=R0,
            P_amb=self.P_AMB,
        )

    def _hoff(self, d_s):
        """Return (omega0^2, beta) with the Hoff et al. (2000) thin-shell terms."""
        P_gas0 = self.P_AMB + 2.0 * self.SIGMA / R0
        k = 3.0 * self.GAMMA * P_gas0 - 2.0 * self.SIGMA / R0
        k_shell = 12.0 * self.G_S * d_s / R0
        c_shell = 12.0 * self.MU_S * d_s / R0
        w2 = (k + k_shell) / (self.RHO * R0**2)
        beta = (4.0 * self.MU + c_shell) / (2.0 * self.RHO * R0**2)
        return w2, beta

    def test_elastic_zero_at_equilibrium(self):
        shell = ThickShell(sigma=0.04, d_s=15e-9, G_s=10e6, mu_s=0.5)
        s = _make_state(1.0)
        assert float(shell.p_elastic(s)) == 0.0

    def test_elastic_formula(self):
        # Qin & Ferrara (2010), Eq. 10.
        d_s, G_s = 100e-9, 11.7e6
        shell = ThickShell(sigma=0.04, d_s=d_s, G_s=G_s, mu_s=0.45)
        V_s = _shell_volume(d_s)
        for x in (0.7, 1.3, 2.0):
            R = x * R0
            expected = (4.0 / 3.0) * G_s * (1.0 - (R0 / R) ** 3) * V_s / (R**3 - V_s)
            assert float(shell.p_elastic(_make_state(x))) == pytest.approx(
                expected, rel=1e-10
            )

    def test_viscous_formula(self):
        d_s, mu_s, R_dot = 100e-9, 0.45, 0.4
        shell = ThickShell(sigma=0.04, d_s=d_s, G_s=11.7e6, mu_s=mu_s)
        V_s = _shell_volume(d_s)
        for x in (0.7, 1.1, 2.0):
            R = x * R0
            expected = 4.0 * mu_s * V_s / (R**3 - V_s) * R_dot / R
            assert float(shell.p_viscous(_make_state(x, R_dot=R_dot))) == pytest.approx(
                expected, rel=1e-10
            )

    @pytest.mark.parametrize("x", [0.6, 0.9, 1.2, 2.5])
    def test_stresses_match_the_integrated_shell_stress(self, x):
        # Independent check of the closed form: integrate the radial stress of
        # an incompressible Kelvin-Voigt shell, tau_rr(r) = -(4 G / (3 r^3))
        # (R^3 - R0^3) - 4 mu R^2 R_dot / r^3 (Qin & Ferrara 2010, Eq. 7), from
        # the inner radius to R. The shell pushes inward with -3 int tau_rr / r.
        d_s, G_s, mu_s, R_dot = 100e-9, 11.7e6, 0.45, 3.0
        shell = ThickShell(sigma=0.04, d_s=d_s, G_s=G_s, mu_s=mu_s)
        R = x * R0
        R1 = (R**3 - _shell_volume(d_s)) ** (1.0 / 3.0)
        r = np.linspace(R1, R, 200_001)
        tau_rr = -(4.0 * G_s / (3.0 * r**3)) * (R**3 - R0**3) - (
            4.0 * mu_s * R**2 * R_dot / r**3
        )
        expected = -3.0 * np.trapezoid(tau_rr / r, r)
        state = _make_state(x, R_dot=R_dot)
        total = float(shell.p_elastic(state) + shell.p_viscous(state))
        assert total == pytest.approx(expected, rel=1e-8)

    @pytest.mark.parametrize("d_s", [1e-9, 15e-9, 100e-9])
    def test_linear_resonance_and_damping_are_exact(self, d_s):
        # Linearising Eq. 10 gives a shell stiffness 4 G_s V_s / R10^3 per unit
        # strain and a shell damping 4 mu_s V_s / R10^3 per unit strain rate,
        # where R10 = R0 - d_s is the inner radius.
        w2, beta = _linear_coefficients(self._eom(d_s))
        factor = _shell_volume(d_s) / (R0 - d_s) ** 3
        P_gas0 = self.P_AMB + 2.0 * self.SIGMA / R0
        k = 3.0 * self.GAMMA * P_gas0 - 2.0 * self.SIGMA / R0
        expected_w2 = (k + 4.0 * self.G_S * factor) / (self.RHO * R0**2)
        expected_beta = (4.0 * self.MU + 4.0 * self.MU_S * factor) / (
            2.0 * self.RHO * R0**2
        )
        assert w2 == pytest.approx(expected_w2, rel=1e-9)
        assert beta == pytest.approx(expected_beta, rel=1e-9)

    @pytest.mark.parametrize("thickness_ratio", [1e-2, 1e-3, 1e-4])
    def test_thin_shell_limit_is_hoff(self, thickness_ratio):
        # Hoff et al. (2000): 12 G_s d_s / R0 and 12 mu_s d_s / R0^2. The
        # relative shell corrections are 2 d_s / R0 + O((d_s / R0)^2).
        d_s = thickness_ratio * R0
        w2, beta = _linear_coefficients(self._eom(d_s))
        w2_hoff, beta_hoff = self._hoff(d_s)
        assert abs(w2 / w2_hoff - 1.0) < 2.5 * thickness_ratio
        assert abs(beta / beta_hoff - 1.0) < 2.5 * thickness_ratio
        # The shell terms themselves converge at first order.
        w2_gas, beta_gas = self._hoff(0.0)
        shell_w2 = (w2 - w2_gas) / (w2_hoff - w2_gas)
        shell_beta = (beta - beta_gas) / (beta_hoff - beta_gas)
        assert shell_w2 - 1.0 == pytest.approx(2.0 * thickness_ratio, rel=0.02)
        assert shell_beta - 1.0 == pytest.approx(2.0 * thickness_ratio, rel=0.02)

    def test_resonance_matches_hoff_for_a_15_nm_shell(self):
        # Regression for the factor-3 error in jbubble 0.1, whose shell
        # stiffness was 4 G_s d_s / R0. With d_s / R0 = 0.0075 the exact
        # model sits within 2 d_s / R0 = 1.5% of Hoff's shell stiffness.
        d_s = 15e-9
        w2, beta = _linear_coefficients(self._eom(d_s))
        w2_hoff, beta_hoff = self._hoff(d_s)
        assert w2 == pytest.approx(w2_hoff, rel=0.015)
        assert beta == pytest.approx(beta_hoff, rel=0.015)

    def test_viscous_term_tends_to_hoff_at_finite_strain(self):
        # The viscous terms agree exactly in the thin-shell limit, at any R.
        mu_s, R_dot = 0.45, 2.0
        for x in (0.7, 1.5):
            R = x * R0
            hoff = 12.0 * mu_s * R0**2 * R_dot / R**4
            errs = []
            for ratio in (1e-3, 1e-4, 1e-5):
                d_s = ratio * R0
                shell = ThickShell(sigma=0.04, d_s=d_s, G_s=11.7e6, mu_s=mu_s)
                p = float(shell.p_viscous(_make_state(x, R_dot=R_dot)))
                errs.append(abs(p / (hoff * d_s) - 1.0))
            assert errs[0] > errs[1] > errs[2]
            assert errs[-1] < 1e-4

    def test_equilibrium_is_at_rest(self):
        eom = self._eom(100e-9)
        y0 = eom.initial_state()
        d = eom(jnp.asarray(0.0), y0, lambda t: 0.0 * t)
        assert float(d.R_dot) == 0.0

    def test_no_callback_in_jaxpr(self):
        def total(d_s, G_s, mu_s):
            shell = ThickShell(sigma=0.04, d_s=d_s, G_s=G_s, mu_s=mu_s)
            return shell(_make_state(1.1, R_dot=0.5))

        assert not _has_callback(total, 100e-9, 11.7e6, 0.45)
        grads = jax.grad(total, argnums=(0, 1, 2))(100e-9, 11.7e6, 0.45)
        assert all(np.isfinite(float(g)) for g in grads)


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

    # x = 1.041 lies between the rupture radius from chi(state), 1.0398 R0,
    # and the one from chi(R0), 1.0422 R0.
    @pytest.mark.parametrize("x", [0.97, 1.005, 1.03, 1.041, 1.2])
    def test_state_dependent_chi_matches_the_closed_form(self, x):
        # chi(state) sets both the elastic branch and the rupture radius.
        st = MarmottantSurfaceTension(
            R_buckle_ratio=0.98,
            chi=RadiusScaledChi(chi0=0.55),
            sigma_rupture=ConstantProperty(SIGMA_R),
        )
        chi = 0.55 * x
        x_r = 0.98 * math.sqrt(1.0 + SIGMA_R / chi)
        if x <= 0.98:
            expected = 0.0
        elif x < x_r:
            expected = chi * ((x / 0.98) ** 2 - 1.0)
        else:
            expected = SIGMA_R
        assert float(st(_make_state(x))) == pytest.approx(expected, rel=1e-9, abs=0.0)

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

    @pytest.mark.parametrize(
        "smoothing,sigma_r",
        [(0.005, SIGMA_R), (0.01, SIGMA_R), (0.05, SIGMA_R), (0.01, 0.05)],
    )
    def test_max_error_is_chi_independent(self, smoothing, sigma_r):
        # sup |sigma - sigma_Marmottant| = smoothing * ln 2 * sigma_r for every
        # chi and ratio: the corner width scales with sigma_r.
        bound = smoothing * math.log(2.0) * sigma_r
        for chi in CHIS:
            for ratio in RATIOS:
                kw = dict(R_buckle_ratio=ratio, chi=float(chi), sigma_rupture=sigma_r)
                x_b = ratio
                x_r = ratio * math.sqrt(1.0 + sigma_r / chi)
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

    @pytest.mark.parametrize(
        "name,value",
        [
            ("R_buckle_ratio", 0.0),
            ("R_buckle_ratio", -0.98),
            ("chi", 0.0),
            ("chi", -0.55),
            ("sigma_rupture", 0.0),
            ("sigma_rupture", -SIGMA_R),
        ],
    )
    def test_non_positive_parameter_raises(self, name, value):
        kw = dict(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=SIGMA_R)
        kw[name] = value
        with pytest.raises(ValueError, match=f"{name} > 0"):
            SmoothMarmottantSurfaceTension(**kw)

    def test_batched_parameters_are_checked(self):
        SmoothMarmottantSurfaceTension(
            R_buckle_ratio=0.98, chi=jnp.asarray([0.3, 0.55]), sigma_rupture=SIGMA_R
        )
        with pytest.raises(ValueError, match="chi > 0"):
            SmoothMarmottantSurfaceTension(
                R_buckle_ratio=0.98,
                chi=jnp.asarray([0.3, -0.55]),
                sigma_rupture=SIGMA_R,
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
                medium=NewtonianMedium(mu=1e-3, rho_L=998.0, c_L=1500.0),
                R0=R0,
                P_amb=101325.0,
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

    def test_sigma_0_equal_to_sigma_rupture_raises(self):
        # chi * ((1 / 0.5)**2 - 1) = 3 * 0.03125 = 0.09375 exactly in binary
        # floating point, so sigma_0 equals sigma_rupture.
        with pytest.raises(ValueError, match="strictly between"):
            GompertzSurfaceTension(
                R_buckle_ratio=0.5, chi=0.03125, sigma_rupture=0.09375
            )

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

    @pytest.mark.parametrize("x", [0.95, 1.02, 1.2])
    def test_state_dependent_chi_matches_the_closed_form(self, x):
        # chi(state) sets b, c, and sigma_0 in Eqs. 13-15 at the current state.
        st = GompertzSurfaceTension(
            R_buckle_ratio=0.98,
            chi=RadiusScaledChi(chi0=0.55),
            sigma_rupture=ConstantProperty(SIGMA_R),
        )
        chi = 0.55 * x
        c = (2 * chi * math.e / SIGMA_R) * math.sqrt(1 + SIGMA_R / (2 * chi))
        sigma_0 = chi * ((1.0 / 0.98) ** 2 - 1.0)
        b = -math.log(sigma_0 / SIGMA_R) / math.exp(c * (1.0 - 1.0 / 0.98))
        expected = SIGMA_R * math.exp(-b * math.exp(c * (1.0 - x / 0.98)))
        assert float(st(_make_state(x))) == pytest.approx(expected, rel=1e-9, abs=0.0)

    @pytest.mark.parametrize(
        "chi,ratio,x",
        [
            # Rejected solver trial steps can reach R < 0, for example
            # R / R0 = -14.6 in a Keller-Miksis run.
            (
                2.0,
                0.995,
                np.concatenate(
                    [np.linspace(-15.0, 0.0, 301), np.linspace(0.01, 10.0, 2001)]
                ),
            ),
            # A stiff shell overflows the uncapped exponent at R > 0 too.
            (10.0, 0.9975, np.asarray([0.01, 0.1])),
        ],
    )
    def test_gradients_finite_under_deep_compression(self, chi, ratio, x):
        # Without the cap on the inner exponent, exp() overflows at these
        # states and both gradients are NaN.
        def sigma(R, chi):
            st = GompertzSurfaceTension(
                R_buckle_ratio=ratio, chi=chi, sigma_rupture=SIGMA_R
            )
            return st(BubbleState(R=R, R0=jnp.asarray(R0)))

        grads = jax.vmap(jax.grad(sigma, argnums=(0, 1)), in_axes=(0, None))(
            jnp.asarray(x * R0), chi
        )
        for g in grads:
            assert bool(jnp.all(jnp.isfinite(g)))
        values = jax.vmap(sigma, in_axes=(0, None))(jnp.asarray(x * R0), chi)
        assert float(values[0]) == 0.0
