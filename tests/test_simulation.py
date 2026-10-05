"""Tests for jbubble.simulation."""

import jax
import jax.numpy as jnp
import pytest
from jbubble import SaveSpec, run_simulation
from jbubble.bubble.state import BubbleState
from jbubble.simulation import SimulationResult


class TestRunSimulation:
    def test_returns_simulation_result(self, simple_eom, sine_pulse):
        result = jax.jit(run_simulation)(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )
        assert isinstance(result, SimulationResult)

    def test_output_shapes(self, simple_eom, sine_pulse):
        N = 300
        result = jax.jit(run_simulation)(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=N),
            t_max=5e-6,
        )
        assert result.ts.shape == (N,)
        assert result.state.R.shape == (N,)
        assert result.state.R_dot.shape == (N,)
        assert result.state_dot.R_dot.shape == (N,)
        assert result.driving_pressure.shape == (N,)

    def test_converges(self, simple_eom, sine_pulse):
        result = jax.jit(run_simulation)(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )
        assert bool(result.converged)

    def test_initial_radius_is_R0(self, simple_eom, sine_pulse):
        result = jax.jit(run_simulation)(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )
        assert float(result.radius[0]) == pytest.approx(simple_eom.R0, rel=1e-4)

    def test_bubble_oscillates(self, simple_eom, sine_pulse):
        result = jax.jit(run_simulation)(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=500),
            t_max=10e-6,
        )
        R_max = float(result.radius.max())
        R_min = float(result.radius.min())
        assert R_max > simple_eom.R0  # expansion
        assert R_min < simple_eom.R0  # compression


class TestSimulationResultAccessors:
    @pytest.fixture
    def result(self, simple_eom, sine_pulse):
        return jax.jit(run_simulation)(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )

    def test_radius(self, result):
        assert jnp.allclose(result.radius, result.state.R)

    def test_radial_velocity(self, result):
        assert jnp.allclose(result.radial_velocity, result.state.R_dot)

    def test_radial_acceleration(self, result):
        assert jnp.allclose(result.radial_acceleration, result.state_dot.R_dot)


class TestInitialState:
    def test_partial_state0_is_filled_from_the_eom(self, simple_eom, sine_pulse):
        R0 = simple_eom.R0
        result = run_simulation(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(100),
            t_max=5e-6,
            state0=BubbleState(R=jnp.asarray(1.2 * R0)),
        )
        assert bool(result.converged)
        assert float(result.radius[0]) == pytest.approx(1.2 * R0, rel=1e-14)
        assert float(result.state.R0[0]) == R0
        expected = simple_eom.initial_state(R=1.2 * R0)
        assert float(result.state.P_gas0[0]) == pytest.approx(
            float(expected.P_gas0), rel=1e-15
        )
