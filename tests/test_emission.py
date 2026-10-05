"""Tests for jbubble.acoustics.emission."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jbubble import SaveSpec, SolverConfig, run_simulation
from jbubble.acoustics.emission import (
    EmissionModel,
    IncompressibleMonopole,
    QuasiAcoustic,
)
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine

RHO_L = 998.0
C_L = 1500.0


def _simulate(num_samples, pressure=100e3, t_max=5e-6, config=None):
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=1e-3),
        R0=2e-6,
        P_amb=101325.0,
        rho_L=RHO_L,
        c_L=C_L,
    )
    pulse = ToneBurst(freq=1e6, pressure=pressure, shape=Sine(), cycle_num=3)
    return eqx.filter_jit(run_simulation)(
        eom,
        pulse,
        save_spec=SaveSpec(num_samples=num_samples),
        t_max=t_max,
        config=config,
    )


@pytest.fixture(scope="module")
def simulation_result():
    """Short simulation for the shape and formula tests."""
    return _simulate(200)


@pytest.fixture(scope="module")
def dense_result():
    """Densely sampled, tightly solved simulation for the derivative checks."""
    tight = SolverConfig(
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
        max_steps=100_000,
    )
    return _simulate(20_001, config=tight)


class TestIncompressibleMonopole:
    def test_output_shape(self, simulation_result):
        emission = IncompressibleMonopole(rho_L=RHO_L)
        p_rad = emission(simulation_result, r=1e-2)
        assert p_rad.shape == (200,)

    def test_equals_volume_acceleration(self, dense_result):
        """p r / rho = d(R^2 Rdot)/dt, with the derivative taken numerically.

        This uses only the saved R and Rdot, not the equation of motion's
        acceleration that the model itself uses.
        """
        r = 1e-2
        p_rad = np.asarray(IncompressibleMonopole(rho_L=RHO_L)(dense_result, r))
        ts = np.asarray(dense_result.ts)
        flux = np.asarray(dense_result.state.R**2 * dense_result.state.R_dot)
        expected = RHO_L / r * np.gradient(flux, ts, edge_order=2)
        scale = np.max(np.abs(expected))
        assert np.max(np.abs(p_rad - expected)) < 1e-4 * scale

    def test_time_integral_recovers_volume_flux(self, dense_result):
        """(r / rho) integral_0^t p dt' = R^2 Rdot(t) - R^2 Rdot(0) (trapezoid)."""
        r = 5e-3
        p_rad = np.asarray(IncompressibleMonopole(rho_L=RHO_L)(dense_result, r))
        ts = np.asarray(dense_result.ts)
        flux = np.asarray(dense_result.state.R**2 * dense_result.state.R_dot)
        dt = np.diff(ts)
        integral = np.concatenate(
            [[0.0], np.cumsum(0.5 * (p_rad[1:] + p_rad[:-1]) * dt)]
        )
        assert np.max(np.abs(r / RHO_L * integral - (flux - flux[0]))) < 1e-4 * np.max(
            np.abs(flux)
        )

    def test_inversely_proportional_to_r(self, simulation_result):
        emission = IncompressibleMonopole(rho_L=RHO_L)
        p1 = emission(simulation_result, r=1e-2)
        p2 = emission(simulation_result, r=2e-2)
        assert jnp.allclose(p1, 2.0 * p2, rtol=1e-12, atol=0.0)

    def test_observer_time_is_emission_time(self, simulation_result):
        emission = IncompressibleMonopole(rho_L=RHO_L)
        t_obs = emission.observer_time(simulation_result, 1e-2)
        assert jnp.array_equal(t_obs, simulation_result.ts)

    def test_vmap_over_distances(self, simulation_result):
        emission = IncompressibleMonopole(rho_L=RHO_L)
        distances = jnp.array([1e-3, 5e-3, 1e-2])
        p_all = jax.vmap(lambda r: emission(simulation_result, r))(distances)
        assert p_all.shape == (3, 200)


class TestQuasiAcoustic:
    def test_is_an_emission_model(self):
        assert isinstance(QuasiAcoustic(rho_L=RHO_L, c_L=C_L), EmissionModel)

    def test_values_equal_monopole(self, simulation_result):
        """The delayed series carries the monopole values unchanged."""
        r = 1.37e-3  # not a whole number of samples of delay
        p_quasi = QuasiAcoustic(rho_L=RHO_L, c_L=C_L)(simulation_result, r)
        p_mono = IncompressibleMonopole(rho_L=RHO_L)(simulation_result, r)
        assert jnp.array_equal(p_quasi, p_mono)

    def test_observer_time_is_shifted_by_travel_time(self, simulation_result):
        r = 1e-2
        t_obs = QuasiAcoustic(rho_L=RHO_L, c_L=C_L).observer_time(simulation_result, r)
        assert jnp.allclose(t_obs, simulation_result.ts + r / C_L, rtol=0, atol=1e-18)

    def test_peak_is_preserved_at_any_distance(self):
        """Resampling R, Rdot, Rddot at retarded times moved the collapse peak.

        The shifted axis keeps every saved sample, so the peak pressure times r
        is the same at every distance.
        """
        result = _simulate(2048, pressure=300e3, t_max=10e-6)
        model = QuasiAcoustic(rho_L=RHO_L, c_L=C_L)
        peaks = [float(jnp.max(model(result, r)) * r) for r in (1e-3, 1.37e-3, 1e-2)]
        assert peaks[0] == pytest.approx(peaks[1], rel=1e-12)
        assert peaks[0] == pytest.approx(peaks[2], rel=1e-12)

    def test_vmap_over_distances(self, simulation_result):
        emission = QuasiAcoustic(rho_L=RHO_L, c_L=C_L)
        distances = jnp.array([1e-3, 5e-3, 1e-2])
        p_all = jax.vmap(lambda r: emission(simulation_result, r))(distances)
        t_all = jax.vmap(lambda r: emission.observer_time(simulation_result, r))(
            distances
        )
        assert p_all.shape == (3, 200)
        assert t_all.shape == (3, 200)
        assert jnp.allclose(t_all[:, 0], distances / C_L)
