"""Tests for jbubble.acoustics.emission."""

import jax
import jax.numpy as jnp
import pytest

from jbubble import SaveSpec, run_simulation
from jbubble.acoustics.emission import IncompressibleMonopole, QuasiAcoustic
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine


@pytest.fixture
def simulation_result():
    """Run a short simulation to produce a result for emission tests."""
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=1e-3),
        R0=2e-6,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )
    pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=3)
    return jax.jit(run_simulation)(
        eom, pulse, save_spec=SaveSpec(num_samples=200), t_max=5e-6
    )


class TestIncompressibleMonopole:
    def test_output_shape(self, simulation_result):
        emission = IncompressibleMonopole(rho_L=998.0)
        p_rad = emission(simulation_result, r=1e-2)
        assert p_rad.shape == (200,)

    def test_formula(self, simulation_result):
        emission = IncompressibleMonopole(rho_L=998.0)
        r = 1e-2
        p_rad = emission(simulation_result, r)

        # Manually compute the expected result
        R = simulation_result.state.R
        R_dot = simulation_result.state.R_dot
        R_ddot = simulation_result.state_dot.R_dot
        expected = 998.0 / r * (2.0 * R * R_dot**2 + R**2 * R_ddot)
        assert jnp.allclose(p_rad, expected, rtol=1e-10)

    def test_inversely_proportional_to_r(self, simulation_result):
        emission = IncompressibleMonopole(rho_L=998.0)
        p1 = emission(simulation_result, r=1e-2)
        p2 = emission(simulation_result, r=2e-2)
        ratio = jnp.max(jnp.abs(p1)) / jnp.max(jnp.abs(p2))
        assert float(ratio) == pytest.approx(2.0, rel=1e-4)

    def test_vmap_over_distances(self, simulation_result):
        emission = IncompressibleMonopole(rho_L=998.0)
        distances = jnp.array([1e-3, 5e-3, 1e-2])
        p_all = jax.vmap(lambda r: emission(simulation_result, r))(distances)
        assert p_all.shape == (3, 200)


class TestQuasiAcoustic:
    def test_output_shape(self, simulation_result):
        emission = QuasiAcoustic(rho_L=998.0, c_L=1500.0)
        p_rad = emission(simulation_result, r=1e-2)
        assert p_rad.shape == (200,)

    def test_finite_output(self, simulation_result):
        emission = QuasiAcoustic(rho_L=998.0, c_L=1500.0)
        p_rad = emission(simulation_result, r=1e-2)
        assert jnp.all(jnp.isfinite(p_rad))

    def test_very_small_r_approaches_monopole(self, simulation_result):
        """At very small r (delay → 0), QuasiAcoustic should approach IncompressibleMonopole."""
        monopole = IncompressibleMonopole(rho_L=998.0)
        quasi = QuasiAcoustic(rho_L=998.0, c_L=1500.0)
        r = 1e-6  # tiny distance → negligible delay

        p_mono = monopole(simulation_result, r)
        p_quasi = quasi(simulation_result, r)

        # Should be close (not exact due to interpolation)
        peak_mono = jnp.max(jnp.abs(p_mono))
        peak_quasi = jnp.max(jnp.abs(p_quasi))
        assert float(peak_quasi) == pytest.approx(float(peak_mono), rel=0.1)

    def test_vmap_over_distances(self, simulation_result):
        emission = QuasiAcoustic(rho_L=998.0, c_L=1500.0)
        distances = jnp.array([1e-3, 5e-3, 1e-2])
        p_all = jax.vmap(lambda r: emission(simulation_result, r))(distances)
        assert p_all.shape == (3, 200)
