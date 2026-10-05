"""Tests for jbubble.simulation."""

import warnings

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jbubble import SaveSpec, SolverConfig, run_simulation
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


class TestConvergenceWarning:
    """`run_simulation` warns about a failed solve only when it can see the flag."""

    _TOO_FEW_STEPS = SolverConfig(max_steps=20)

    def test_warns_when_called_eagerly(self, simple_eom, sine_pulse):
        with pytest.warns(UserWarning, match="did not converge"):
            result = run_simulation(
                simple_eom,
                sine_pulse,
                save_spec=SaveSpec(50),
                t_max=5e-6,
                config=self._TOO_FEW_STEPS,
            )
        assert not bool(result.converged)
        assert bool(jnp.isinf(result.radius[-1]))

    def test_silent_and_callback_free_under_jit(self, simple_eom, sine_pulse):
        def simulate(eom, pulse):
            return run_simulation(
                eom,
                pulse,
                save_spec=SaveSpec(50),
                t_max=5e-6,
                config=self._TOO_FEW_STEPS,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = jax.jit(simulate)(simple_eom, sine_pulse)
        assert not bool(result.converged)
        # diffrax's own `eqx.error_if` adds a `pure_callback` that runs only if
        # it raises; jbubble adds no `debug_callback`.
        jaxpr = str(jax.make_jaxpr(simulate)(simple_eom, sine_pulse))
        assert "debug_callback" not in jaxpr

    def test_no_warning_when_converged(self, simple_eom, sine_pulse):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_simulation(simple_eom, sine_pulse, save_spec=SaveSpec(50), t_max=2e-6)


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


class TestAdjoint:
    def test_forward_mode_jacobian_matches_reverse_mode(self, simple_eom, sine_pulse):
        def peak(mu, adjoint=None):
            eom = eqx.tree_at(lambda e: e.medium.mu.val, simple_eom, mu)
            result = run_simulation(
                eom, sine_pulse, save_spec=SaveSpec(200), t_max=5e-6, adjoint=adjoint
            )
            return jnp.max(result.radius)

        mu = jnp.asarray(1e-3)
        g_forward = jax.jit(jax.jacfwd(lambda m: peak(m, diffrax.ForwardMode())))(mu)
        g_reverse = jax.jit(jax.grad(peak))(mu)
        assert bool(jnp.isfinite(g_forward)) and float(g_forward) < 0.0
        assert jnp.allclose(g_forward, g_reverse, rtol=1e-6)
