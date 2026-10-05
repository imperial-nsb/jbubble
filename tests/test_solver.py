"""Tests for jbubble.solver."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jbubble.solver as solver_module
import numpy as np
import optimistix as optx
import pytest
from jbubble.bubble.eom import Gilmore, KellerMiksis
from jbubble.bubble.gas import PolytropicGas, VanDerWaalsGas
from jbubble.bubble.medium import KelvinVoigtMedium, NewtonianMedium
from jbubble.bubble.shell import LipidShell, MarmottantSurfaceTension, NoShell
from jbubble.bubble.state import BubbleState
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.simulation import run_simulation
from jbubble.solver import SaveSpec, SolverConfig, _guarded_vector_field, solve_eom

_PID = diffrax.PIDController
# Loose tolerances: rejected trial steps leave the domain during a collapse.
_LOOSE = SolverConfig(stepsize_controller=_PID(rtol=1e-4, atol=1e-6))
# The jbubble 0.1 fit_parameters tolerances.
_FIT01 = SolverConfig(stepsize_controller=_PID(rtol=1e-4, atol=1e-8))
_VERY_LOOSE = SolverConfig(stepsize_controller=_PID(rtol=1e-3, atol=1e-6))
_TIGHT = SolverConfig(stepsize_controller=_PID(rtol=1e-8, atol=1e-10))


def _unguarded_vector_field(t, z, args):
    """`_guarded_vector_field` without the guard."""
    eom, pulse, _, scale = args
    state = jtu.tree_map(jnp.multiply, z, scale)
    return jtu.tree_map(jnp.divide, eom(t, state, pulse), scale)


def _scaled_args(eom, pulse):
    """Return the `_guarded_vector_field` arguments that `solve_eom` builds."""
    y0 = eom.initial_state()
    scale = eom.state_scale(y0)
    return (eom, pulse, jtu.tree_map(jnp.divide, y0, scale), scale)


def _scaled(state, scale):
    return jtu.tree_map(jnp.divide, state, scale)


def _marmottant(kappa_s):
    sigma = MarmottantSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
    return LipidShell(sigma=sigma, kappa_s=kappa_s)


def _lipid_bubble(kappa_s, *, gamma=1.07, R0=2e-6):
    return KellerMiksis(
        gas=PolytropicGas(gamma=gamma),
        shell=_marmottant(kappa_s),
        medium=NewtonianMedium(mu=1e-3),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


def _free_bubble(mu, *, R0=2e-6):
    return KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=mu),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


def _vdw_bubble(mu):
    return KellerMiksis(
        gas=VanDerWaalsGas(gamma=1.4, h_frac=1 / 5.61),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=mu),
        R0=2e-6,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


def _gilmore_bubble(kappa_s):
    return Gilmore(
        gas=PolytropicGas(gamma=1.07),
        shell=_marmottant(kappa_s),
        medium=NewtonianMedium(mu=1e-3),
        R0=2e-6,
        P_amb=101325.0,
        rho_L=998.0,
    )


def _tissue_bubble(G):
    return KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.056),
        medium=KelvinVoigtMedium(mu=0.015, G=G),
        R0=1e-6,
        P_amb=101325.0,
        rho_L=1060.0,
        c_L=1540.0,
    )


def _tone(pressure=100e3, cycles=5, freq=1e6):
    return ToneBurst(freq=freq, pressure=pressure, shape=Sine(), cycle_num=cycles)


def _radius_loss(make_eom, pulse, *, t_max=10e-6, config=None, adjoint=None):
    """Return `p -> (mean((R/R0 - 1)^2), solution)` for `jax.value_and_grad`."""

    def loss(p):
        sol = solve_eom(
            make_eom(p),
            pulse,
            t_max=t_max,
            save_spec=SaveSpec(500),
            config=config,
            adjoint=adjoint,
        )
        return jnp.mean((sol.ys.R / sol.ys.R0 - 1.0) ** 2), sol

    return loss


class TestSaveSpec:
    def test_default_num_samples(self):
        spec = SaveSpec()
        assert spec.num_samples == 1024

    def test_custom_num_samples(self):
        spec = SaveSpec(num_samples=500)
        assert spec.num_samples == 500

    def test_build(self):
        spec = SaveSpec(num_samples=100)
        saveat = spec.build(jnp.asarray(0.0), jnp.asarray(10e-6))
        assert isinstance(saveat, diffrax.SaveAt)


class TestSolverConfig:
    def test_defaults(self):
        config = SolverConfig()
        assert isinstance(config.solver, diffrax.Dopri5)
        assert isinstance(config.stepsize_controller, diffrax.PIDController)
        assert config.stepsize_controller.rtol == 1e-6
        assert config.stepsize_controller.atol == 1e-10
        assert config.dt0 == 1e-9
        assert config.max_steps == 100_000

    def test_custom_config(self):
        config = SolverConfig(
            solver=diffrax.Kvaerno5(),
            dt0=1e-10,
            max_steps=50_000,
        )
        assert isinstance(config.solver, diffrax.Kvaerno5)
        assert config.dt0 == 1e-10
        assert config.max_steps == 50_000

    def test_stiff_uses_kvaerno5_with_chord_root_finder(self):
        config = SolverConfig.stiff()
        assert isinstance(config.solver, diffrax.Kvaerno5)
        root_finder = config.solver.root_finder
        assert isinstance(root_finder, optx.Chord)
        assert (root_finder.rtol, root_finder.atol) == (1e-6, 1e-6)
        assert root_finder.norm is optx.rms_norm
        assert config.stepsize_controller.rtol == 1e-6
        assert config.stepsize_controller.atol == 1e-10
        assert config.max_steps == 100_000

    def test_stiff_passes_tolerances_to_controller_and_newton(self):
        config = SolverConfig.stiff(rtol=1e-8, atol=1e-5, dt0=1e-10, max_steps=7)
        assert config.stepsize_controller.rtol == 1e-8
        assert config.stepsize_controller.atol == 1e-5
        assert config.solver.root_finder.rtol == 1e-8
        assert config.solver.root_finder.atol == 1e-5
        assert config.dt0 == 1e-10
        assert config.max_steps == 7

    def test_stiff_floors_the_newton_atol(self):
        config = SolverConfig.stiff(atol=1e-12)
        assert config.stepsize_controller.atol == 1e-12
        assert config.solver.root_finder.atol == 1e-6


class TestSolveEom:
    def test_returns_solution(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=100),
            t_max=5e-6,
        )
        assert isinstance(sol, diffrax.Solution)

    def test_solution_has_correct_shape(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )
        assert sol.ts.shape == (200,)
        assert sol.ys.R.shape == (200,)

    def test_converges(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=100),
            t_max=5e-6,
        )
        assert diffrax.is_successful(sol.result)

    def test_returns_state_in_si_units(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom, sine_pulse, save_spec=SaveSpec(num_samples=100), t_max=5e-6
        )
        y0 = simple_eom.initial_state()
        assert float(sol.ys.R[0]) == float(y0.R)
        assert jnp.all(sol.ys.R0 == y0.R0)
        assert jnp.allclose(sol.ys.P_gas0, y0.P_gas0, rtol=1e-15, atol=0)

    def test_uses_pulse_t_end_when_no_t_max(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=100),
        )
        expected_t_end = float(sine_pulse.t_end)
        assert float(sol.ts[-1]) == pytest.approx(expected_t_end, rel=1e-6)


class TestScaledState:
    """The solver integrates `state / eom.state_scale(state)`."""

    def test_state_scale(self, simple_eom):
        y0 = simple_eom.initial_state()
        scale = simple_eom.state_scale(y0)
        assert float(scale.R) == float(y0.R0)
        assert float(scale.R0) == float(y0.R0)
        assert float(scale.R_dot) == pytest.approx((101325.0 / 998.0) ** 0.5)
        assert float(scale.P_gas0) == float(y0.P_gas0)

    def test_state_scale_falls_back_to_P_amb(self, simple_eom):
        y0 = eqx.tree_at(lambda s: s.P_gas0, simple_eom.initial_state(), 0.0)
        assert float(simple_eom.state_scale(y0).P_gas0) == 101325.0

    def test_frozen_fields_round_trip_exactly(self, simple_eom):
        """`(x / x) * x == x`, so the solver sees `R0` and `P_gas0` exactly."""
        y0 = simple_eom.initial_state(R=1.3 * simple_eom.R0)
        scale = simple_eom.state_scale(y0)
        back = jtu.tree_map(jnp.multiply, _scaled(y0, scale), scale)
        assert float(back.R0) == float(y0.R0)
        assert float(back.P_gas0) == float(y0.P_gas0)

    def test_step_sequence_is_independent_of_bubble_size(self):
        """An inviscid, tension-free Keller-Miksis bubble is self-similar.

        Scaling R0, t, and 1/f by 1/32 (exact in binary floating point) maps
        the problem onto itself, so with tolerances on R/R0 the solver takes
        exactly the same steps for both sizes. With an absolute tolerance in
        metres, it would not.
        """

        def solve(scale):
            eom = KellerMiksis(
                gas=PolytropicGas(gamma=1.4),
                shell=NoShell(sigma=0.0),
                medium=NewtonianMedium(mu=0.0),
                R0=2e-6 * scale,
                P_amb=101325.0,
                rho_L=998.0,
                c_L=1500.0,
            )
            pulse = _tone(150e3, cycles=3, freq=1e6 / scale)
            config = SolverConfig(dt0=1e-9 * scale)
            return solve_eom(
                eom, pulse, t_max=4e-6 * scale, save_spec=SaveSpec(256), config=config
            )

        big, small = solve(1.0), solve(1.0 / 32.0)
        assert int(big.stats["num_steps"]) == int(small.stats["num_steps"])
        assert int(big.stats["num_rejected_steps"]) == int(
            small.stats["num_rejected_steps"]
        )
        assert jnp.allclose(big.ys.R / 2e-6, small.ys.R / (2e-6 / 32), rtol=1e-13)

    def test_matches_an_si_state_solve(self):
        """At tight tolerances the scaled solve matches a solve of the SI state."""
        eom, pulse = _lipid_bubble(5e-9), _tone(200e3)
        config = SolverConfig(stepsize_controller=_PID(rtol=1e-11, atol=1e-13))
        scaled = solve_eom(eom, pulse, save_spec=SaveSpec(400), config=config)
        y0 = eom.initial_state()
        t0, t1 = jnp.asarray(0.0), jnp.asarray(pulse.t_end)
        si = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: eom(t, y, pulse)),
            diffrax.Tsit5(),
            t0,
            t1,
            1e-9,
            y0,
            saveat=SaveSpec(400).build(t0, t1),
            stepsize_controller=_PID(rtol=1e-11, atol=1e-13 * 2e-6),
            max_steps=1_000_000,
        )
        assert float(jnp.max(jnp.abs(scaled.ys.R - si.ys.R))) < 5e-8 * 2e-6
        velocity_error = jnp.max(jnp.abs(scaled.ys.R_dot - si.ys.R_dot))
        assert float(velocity_error) < 1e-6 * float(jnp.max(jnp.abs(si.ys.R_dot)))

    @pytest.mark.slow
    def test_gradients_through_the_scale_match_finite_differences(self):
        """`R0`, `P_amb`, and `rho_L` also set the (stop-gradient) scale."""
        pulse = _tone(150e3)
        config = SolverConfig(stepsize_controller=_PID(rtol=1e-11, atol=1e-13))
        theta0 = np.array([2e-6, 101325.0, 998.0])

        def loss(theta):
            eom = KellerMiksis(
                gas=PolytropicGas(gamma=1.4),
                shell=NoShell(sigma=0.072),
                medium=NewtonianMedium(mu=1e-3),
                R0=theta[0],
                P_amb=theta[1],
                rho_L=theta[2],
                c_L=1500.0,
            )
            sol = solve_eom(eom, pulse, save_spec=SaveSpec(400), config=config)
            return jnp.mean((sol.ys.R / 2e-6) ** 2)

        grad = np.asarray(jax.jit(jax.grad(loss))(jnp.asarray(theta0)))
        f = jax.jit(loss)
        for i in range(3):
            h = theta0[i] * 1e-5
            up, down = theta0.copy(), theta0.copy()
            up[i] += h
            down[i] -= h
            fd = (float(f(jnp.asarray(up))) - float(f(jnp.asarray(down)))) / (2 * h)
            assert grad[i] == pytest.approx(fd, rel=1e-5)


class TestInitialStateFilling:
    def test_partial_state_gets_equilibrium_fields(self, simple_eom, sine_pulse):
        R0 = simple_eom.R0
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            y0=BubbleState(R=jnp.asarray(1.2 * R0)),
            save_spec=SaveSpec(100),
            t_max=5e-6,
        )
        y_eq = simple_eom.initial_state()
        assert diffrax.is_successful(sol.result)
        assert float(sol.ys.R[0]) == pytest.approx(1.2 * R0, rel=1e-14)
        assert jnp.all(sol.ys.R0 == y_eq.R0)
        assert jnp.allclose(sol.ys.P_gas0, y_eq.P_gas0, rtol=1e-15, atol=0)

    def test_matches_initial_state_keywords(self, simple_eom, sine_pulse):
        R0 = simple_eom.R0

        def solve(y0):
            return solve_eom(
                simple_eom, sine_pulse, y0=y0, save_spec=SaveSpec(100), t_max=5e-6
            )

        partial = solve(BubbleState(R=jnp.asarray(1.2 * R0), R_dot=jnp.asarray(0.5)))
        full = solve(simple_eom.initial_state(R=1.2 * R0, R_dot=0.5))
        assert jnp.array_equal(partial.ys.R, full.ys.R)

    def test_explicit_fields_are_kept(self, simple_eom, sine_pulse):
        y0 = BubbleState(
            R=jnp.asarray(3e-6),
            R0=jnp.asarray(3e-6),
            P_gas0=jnp.asarray(2e5),
        )
        sol = solve_eom(
            simple_eom, sine_pulse, y0=y0, save_spec=SaveSpec(10), t_max=1e-7
        )
        assert jnp.all(sol.ys.R0 == 3e-6)
        assert jnp.allclose(sol.ys.P_gas0, 2e5, rtol=1e-15, atol=0)

    def test_filled_P_gas0_uses_the_given_R0(self, simple_eom, sine_pulse):
        """A set R0 with an unset P_gas0 gets the Laplace pressure at that R0."""
        y0 = BubbleState(R=jnp.asarray(3e-6), R0=jnp.asarray(3e-6))
        sol = solve_eom(
            simple_eom, sine_pulse, y0=y0, save_spec=SaveSpec(10), t_max=1e-7
        )
        expected = 101325.0 + 2.0 * 0.072 / 3e-6
        assert jnp.allclose(sol.ys.P_gas0, expected, rtol=1e-14, atol=0)

    def test_integer_zero_is_filled_without_truncation(self, simple_eom, sine_pulse):
        """An integer `R0=0` or `P_gas0=0` fills like a float zero."""
        R = jnp.asarray(1.2 * simple_eom.R0)

        def solve(y0):
            return solve_eom(
                simple_eom, sine_pulse, y0=y0, save_spec=SaveSpec(10), t_max=1e-6
            )

        expected = solve(BubbleState(R=R))
        sol = solve(BubbleState(R=R, R0=0, P_gas0=0))
        assert diffrax.is_successful(sol.result)
        assert jnp.array_equal(sol.ys.R0, expected.ys.R0)
        assert jnp.array_equal(sol.ys.P_gas0, expected.ys.P_gas0)
        assert jnp.array_equal(sol.ys.R, expected.ys.R)

    def test_works_under_jit(self, simple_eom, sine_pulse):
        @jax.jit
        def final_radius(R):
            sol = solve_eom(
                simple_eom,
                sine_pulse,
                y0=BubbleState(R=R),
                save_spec=SaveSpec(10),
                t_max=1e-6,
            )
            return sol.ys.R0[-1], sol.ys.P_gas0[-1]

        R0, P_gas0 = final_radius(jnp.asarray(2.2e-6))
        assert float(R0) == simple_eom.R0
        assert float(P_gas0) == pytest.approx(
            float(simple_eom.initial_state().P_gas0), rel=1e-15
        )


class TestNoHostCallbacks:
    """jbubble adds no host callbacks to a traced solve.

    The jaxpr does contain equinox's `pure_callback` error branch from inside
    diffrax (`eqx.error_if`), which runs only if diffrax raises.
    """

    def test_solve_eom_jaxpr_has_no_debug_callback(self, simple_eom, sine_pulse):
        jaxpr = jax.make_jaxpr(
            lambda e, p: solve_eom(e, p, save_spec=SaveSpec(16), t_max=1e-6).ys.R
        )(simple_eom, sine_pulse)
        assert "debug_callback" not in str(jaxpr)

    def test_run_simulation_jaxpr_has_no_debug_callback(self, simple_eom, sine_pulse):
        jaxpr = jax.make_jaxpr(
            lambda e, p: run_simulation(e, p, save_spec=SaveSpec(16), t_max=1e-6).radius
        )(simple_eom, sine_pulse)
        assert "debug_callback" not in str(jaxpr)


# ── gradient-safe right-hand side ────────────────────────────────────────────


def _bitwise_equal(a, b):
    return all(
        bool(jnp.array_equal(x, y))
        for x, y in zip(jtu.tree_leaves(a), jtu.tree_leaves(b), strict=True)
    )


class TestGuardedVectorField:
    @pytest.mark.parametrize(
        ("R", "R_dot", "P_gas0_factor"),
        [
            (-1e-6, -10.0, 1.0),
            (0.0, 0.0, 1.0),
            (jnp.nan, 1.0, 1.0),
            (1e-6, jnp.nan, 1.0),
            (1e-6, 1.0, jnp.nan),  # a NaN stage also reaches P_gas0
            (jnp.inf, 1.0, 1.0),
        ],
    )
    def test_nan_outside_domain_with_finite_vjp(self, R, R_dot, P_gas0_factor):
        eom = _lipid_bubble(1e-9)
        args = _scaled_args(eom, _tone())
        y0 = eom.initial_state()
        y = BubbleState(
            R=jnp.asarray(R),
            R_dot=jnp.asarray(R_dot),
            R0=y0.R0,
            P_gas0=y0.P_gas0 * P_gas0_factor,
        )
        t = jnp.asarray(1e-7)
        out, vjp = jax.vjp(
            lambda z: _guarded_vector_field(t, z, args), _scaled(y, args[3])
        )
        assert all(bool(jnp.isnan(v)) for v in jtu.tree_leaves(out))
        (cotangent,) = vjp(jtu.tree_map(jnp.zeros_like, out))
        assert all(bool(jnp.all(jnp.isfinite(v))) for v in jtu.tree_leaves(cotangent))

    def test_van_der_waals_hard_core_is_outside_domain(self):
        """Inside the hard core (0 < R <= h) the gas law is unphysical, so the
        guard rejects it too."""
        eom = _vdw_bubble(1e-3)
        args = _scaled_args(eom, _tone())
        y0 = eom.initial_state()
        h = float(y0.R0) / 5.61
        y = BubbleState(
            R=jnp.asarray(0.9 * h), R_dot=jnp.asarray(-50.0), R0=y0.R0, P_gas0=y0.P_gas0
        )
        assert not bool(eom.is_admissible(y))
        t = jnp.asarray(1e-7)
        out, vjp = jax.vjp(
            lambda z: _guarded_vector_field(t, z, args), _scaled(y, args[3])
        )
        assert all(bool(jnp.isnan(v)) for v in jtu.tree_leaves(out))
        (cotangent,) = vjp(jtu.tree_map(jnp.zeros_like, out))
        assert all(bool(jnp.all(jnp.isfinite(v))) for v in jtu.tree_leaves(cotangent))
        just_outside = eqx.tree_at(lambda s: s.R, y, jnp.asarray(1.01 * h))
        assert bool(eom.is_admissible(just_outside))

    def test_bitwise_equal_to_unguarded_inside_domain(self):
        """The right-hand side and its derivatives match the unguarded field.

        Every argument is traced, as inside a solve, so XLA can't constant-fold
        the comparison away.
        """
        eom = _lipid_bubble(5e-9)
        args = _scaled_args(eom, _tone(300e3))
        y0 = eom.initial_state()
        rng = np.random.default_rng(0)

        def evaluate(field):
            def all_derivatives(t, z, args, dz):
                def f(zz):
                    return field(t, zz, args)

                return f(z), jax.jvp(f, (z,), (dz,))[1], jax.jacfwd(f)(z)

            return eqx.filter_jit(all_derivatives)

        guarded, plain = (
            evaluate(_guarded_vector_field),
            evaluate(_unguarded_vector_field),
        )
        dz = jtu.tree_map(lambda x: jnp.asarray(rng.normal()), y0)
        for _ in range(50):
            y = BubbleState(
                R=jnp.asarray(2e-6 * rng.uniform(0.2, 3.0)),
                R_dot=jnp.asarray(rng.uniform(-200.0, 200.0)),
                R0=y0.R0,
                P_gas0=y0.P_gas0,
            )
            t = jnp.asarray(rng.uniform(0.0, 5e-6))
            z = _scaled(y, args[3])
            assert _bitwise_equal(guarded(t, z, args, dz), plain(t, z, args, dz))

    @staticmethod
    def _guarded_and_plain(monkeypatch, make_eom, p, pulse, config, t_max):
        def solve():
            return jax.jit(
                lambda p: solve_eom(
                    make_eom(p),
                    pulse,
                    t_max=t_max,
                    save_spec=SaveSpec(500),
                    config=config,
                )
            )(jnp.asarray(p))

        guarded = solve()
        with monkeypatch.context() as patch:
            patch.setattr(
                solver_module, "_guarded_vector_field", _unguarded_vector_field
            )
            plain = solve()
        assert diffrax.is_successful(guarded.result)
        assert int(guarded.stats["num_rejected_steps"]) > 0
        return guarded, plain

    @pytest.mark.parametrize(
        ("make_eom", "p", "pressure", "config", "t_max"),
        [
            (_lipid_bubble, 5e-9, 200e3, SolverConfig(), 10e-6),
            (_lipid_bubble, 7.2e-9, 400e3, _LOOSE, 10e-6),
            (_free_bubble, 1e-3, 300e3, _FIT01, 10e-6),
            (_vdw_bubble, 1e-3, 500e3, _VERY_LOOSE, 10e-6),
            (_gilmore_bubble, 5e-9, 300e3, _LOOSE, 10e-6),
            (_tissue_bubble, 1e6, 1e6, _TIGHT, 6e-6),
        ],
        ids=[
            "lipid-200k-default",
            "lipid-400k-loose",
            "free-300k-fit01",
            "vdw-500k-rtol1e-3",
            "gilmore-300k-loose",
            "tissue-1MPa-tight",
        ],
    )
    def test_explicit_solution_bit_identical(
        self, monkeypatch, make_eom, p, pressure, config, t_max
    ):
        """With Dopri5 the guard changes no accepted step: R(t), R_dot(t),
        and the step counts are equal.

        Every case rejects trial steps; the loose ones reject steps that leave
        the domain, where the unguarded gradient is NaN.
        """
        guarded, plain = self._guarded_and_plain(
            monkeypatch, make_eom, p, _tone(pressure), config, t_max
        )
        assert bool(jnp.array_equal(guarded.ys.R, plain.ys.R))
        assert bool(jnp.array_equal(guarded.ys.R_dot, plain.ys.R_dot))
        for key in ("num_steps", "num_accepted_steps", "num_rejected_steps"):
            assert int(guarded.stats[key]) == int(plain.stats[key])

    @pytest.mark.parametrize(
        ("make_eom", "p", "pulse", "t_max"),
        [
            (_lipid_bubble, 7.2e-9, _tone(400e3), 10e-6),
            (_tissue_bubble, 1e6, _tone(1e6), 6e-6),
            (lambda k: _lipid_bubble(k, R0=50e-9), 7.5e-9, _tone(freq=5e6), 2e-6),
            (lambda mu: _free_bubble(mu, R0=0.3e-6), 0.05, _tone(freq=2e6), 5e-6),
        ],
        ids=["lipid-400k", "tissue-1MPa", "lipid-50nm", "viscous-300nm"],
    )
    def test_implicit_solution_agrees_to_tolerance(
        self, monkeypatch, make_eom, p, pulse, t_max
    ):
        """With Kvaerno5, the guard can change how XLA rounds the Newton
        iterations, and the controller then picks different steps. The
        solutions still agree to within the solver's own error, about
        1e-8 R0 for the 50 nm bubble, but not always bit for bit."""
        guarded, plain = self._guarded_and_plain(
            monkeypatch, make_eom, p, pulse, SolverConfig.stiff(), t_max
        )
        assert diffrax.is_successful(plain.result)
        R0 = float(guarded.ys.R0[0])
        assert float(jnp.max(jnp.abs(guarded.ys.R - plain.ys.R))) < 1e-7 * R0

    def test_guard_fixes_nan_gradient_and_keeps_the_forward_pass(self, monkeypatch):
        """Rejected trial steps reach R <= 0. Without the guard the gradient is
        NaN; with it, it's finite, and the forward pass inside `jax.grad` is
        unchanged."""
        loss = _radius_loss(_lipid_bubble, _tone(200e3), config=_LOOSE)

        def value_and_grad():
            (value, sol), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(
                jnp.asarray(5e-9)
            )
            return value, sol.ys.R, grad

        value, R, grad = value_and_grad()
        monkeypatch.setattr(
            solver_module, "_guarded_vector_field", _unguarded_vector_field
        )
        plain_value, plain_R, plain_grad = value_and_grad()
        assert bool(jnp.isnan(plain_grad))
        assert bool(jnp.isfinite(grad))
        assert float(value) == float(plain_value)
        assert bool(jnp.array_equal(R, plain_R))


class TestGradients:
    """Finite gradients through strong collapses at the default tolerances.

    These are regression tests for the defaults, not for the guard: at these
    tolerances no rejected stage leaves the admissible domain.
    `TestGuardedVectorField` covers the guard with looser tolerances.
    """

    def test_keller_miksis_marmottant_400kpa(self):
        loss = _radius_loss(_lipid_bubble, _tone(400e3))
        (value, sol), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(
            jnp.asarray(7.2e-9)
        )
        assert diffrax.is_successful(sol.result)
        assert bool(jnp.isfinite(value)) and bool(jnp.isfinite(grad))

    @pytest.mark.parametrize("pressure", [200e3, 350e3, 500e3])
    def test_van_der_waals_strong_drive(self, pressure):
        loss = _radius_loss(_vdw_bubble, _tone(pressure))
        (value, sol), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(
            jnp.asarray(1e-3)
        )
        assert diffrax.is_successful(sol.result)
        assert bool(jnp.isfinite(value)) and bool(jnp.isfinite(grad))

    @pytest.mark.slow
    def test_inertial_collapse_gradient_matches_tight_reference(self):
        """At the default tolerances the gradient through an inertial collapse
        (Rmax/R0 about 4.4) is within 2 % of the converged gradient."""
        pulse = _tone(300e3)
        tight = SolverConfig(stepsize_controller=_PID(rtol=1e-11, atol=1e-13))

        def grad(config):
            loss = _radius_loss(_free_bubble, pulse, config=config)
            return float(jax.jit(jax.grad(lambda mu: loss(mu)[0]))(jnp.asarray(1e-3)))

        assert grad(SolverConfig()) == pytest.approx(grad(tight), rel=2e-2)

    @pytest.mark.slow
    def test_reverse_and_forward_mode_agree(self):
        def loss(adjoint):
            return _radius_loss(_lipid_bubble, _tone(400e3), adjoint=adjoint)

        k = jnp.asarray(7.2e-9)
        reverse = loss(diffrax.RecursiveCheckpointAdjoint())
        forward = loss(diffrax.ForwardMode())
        g_reverse = jax.jit(jax.grad(lambda k: reverse(k)[0]))(k)
        g_forward = jax.jit(jax.jacfwd(lambda k: forward(k)[0]))(k)
        assert bool(jnp.isfinite(g_reverse))
        # Both differentiate the same discretised solve; they agree closely at
        # the default tolerances but not bit for bit.
        assert jnp.allclose(g_reverse, g_forward, rtol=1e-5)
