"""Tests for jbubble.simulation."""

import jax
import jax.numpy as jnp
import pytest
from jbubble import SaveSpec, run_simulation
from jbubble.bubble.eom import SphericalConfinement
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.bubble.state import ConfinedBubbleState
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
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

    def test_has_vessel_false(self, result):
        assert not result.has_vessel

    def test_vessel_radius_none(self, result):
        assert result.vessel_radius is None

    def test_vessel_velocity_none(self, result):
        assert result.vessel_velocity is None


class TestConfinedSimulation:
    @pytest.fixture
    def confined_result(self):
        eom = SphericalConfinement(
            gas=PolytropicGas(gamma=1.4),
            shell=NoShell(sigma=0.072),
            medium=NewtonianMedium(mu=1e-3),
            R0=2e-6,
            P_amb=101325.0,
            rho_L=998.0,
            c_L=1500.0,
            vessel_radius=50e-6,
            vessel_rho=1050.0,
            vessel_E=1e6,
            vessel_nu=0.49,
            vessel_d=1e-6,
            tissue_rho=1050.0,
            tissue_d=1e-3,
        )
        pulse = ToneBurst(freq=1e6, pressure=50e3, shape=Sine(), cycle_num=3)
        return jax.jit(run_simulation)(
            eom, pulse, save_spec=SaveSpec(num_samples=200), t_max=5e-6
        )

    def test_has_vessel_true(self, confined_result):
        assert confined_result.has_vessel

    def test_vessel_radius_shape(self, confined_result):
        assert confined_result.vessel_radius.shape == (200,)

    def test_vessel_velocity_shape(self, confined_result):
        assert confined_result.vessel_velocity.shape == (200,)

    def test_state_is_confined(self, confined_result):
        assert isinstance(confined_result.state, ConfinedBubbleState)
