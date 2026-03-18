"""Tests for jbubble.solver."""

import diffrax
import jax.numpy as jnp
import pytest
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
        assert isinstance(config.solver, diffrax.Kvaerno5)
        assert isinstance(config.stepsize_controller, diffrax.PIDController)
        assert config.dt0 == 1e-9
        assert config.max_steps == 10_000

    def test_custom_config(self):
        config = SolverConfig(
            solver=diffrax.Dopri5(),
            dt0=1e-10,
            max_steps=50_000,
        )
        assert isinstance(config.solver, diffrax.Dopri5)
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
