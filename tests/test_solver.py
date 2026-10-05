"""Tests for jbubble.solver."""

import diffrax
import jax
import jax.numpy as jnp
import pytest
from jbubble.bubble.state import BubbleState
from jbubble.simulation import run_simulation
from jbubble.solver import SaveSpec, SolverConfig, solve_eom


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
        assert config.dt0 == 1e-9
        assert config.max_steps == 10_000

    def test_custom_config(self):
        config = SolverConfig(
            solver=diffrax.Kvaerno5(),
            dt0=1e-10,
            max_steps=50_000,
        )
        assert isinstance(config.solver, diffrax.Kvaerno5)
        assert config.dt0 == 1e-10
        assert config.max_steps == 50_000


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

    def test_uses_initial_state_from_eom(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=100),
            t_max=5e-6,
        )
        expected_R0 = simple_eom.R0
        assert float(sol.ys.R[0]) == pytest.approx(expected_R0, rel=1e-4)

    def test_uses_pulse_t_end_when_no_t_max(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=100),
        )
        expected_t_end = float(sine_pulse.t_end)
        assert float(sol.ts[-1]) == pytest.approx(expected_t_end, rel=1e-6)


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
