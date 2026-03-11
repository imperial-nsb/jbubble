"""Tests for jbubble.simulation — high-level simulation runner and results."""

import jax.numpy as jnp
import numpy as np
from jbubble.simulation import (
    SimulationResult,
    arrays_from_result,
    compute_radius_metrics,
    run_simulation,
)

from conftest import make_confinement

# ── run_simulation ────────────────────────────────────────────────────────────


def test_run_simulation_returns_result(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    assert isinstance(result, SimulationResult)


def test_run_simulation_ts_shape(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    assert result.ts.shape == (save_spec.num_samples,)


def test_run_simulation_array_shapes_consistent(km_eom, pulse, units, save_spec):
    result = run_simulation(km_eom, pulse, units=units, save_spec=save_spec)
    n = save_spec.num_samples
    assert result.radius.shape == (n,)
    assert result.radial_velocity.shape == (n,)
    assert result.radial_acceleration.shape == (n,)
    assert result.driving_pressure.shape == (n,)


def test_run_simulation_ts_in_physical_units(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    # Times should be in seconds (order 1e-6 to 1e-4)
    assert bool(jnp.all(result.ts >= 0.0))
    assert float(jnp.max(result.ts)) < 1e-2  # < 10 ms


def test_run_simulation_radius_in_physical_units(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    # Radius should be around 1–10 µm in metres
    assert float(jnp.min(result.radius)) > 0.0
    assert float(jnp.max(result.radius)) < 100e-6  # < 100 µm


def test_run_simulation_no_vessel_for_simple_models(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    assert result.vessel_radius is None
    assert result.vessel_velocity is None
    assert not result.has_vessel


def test_run_simulation_has_vessel_for_confinement(pulse, units, save_spec):
    eom = make_confinement(R0=3e-6)
    result = run_simulation(eom, pulse, units=units, save_spec=save_spec)
    assert result.vessel_radius is not None
    assert result.vessel_velocity is not None
    assert result.has_vessel


# ── radiated_pressure ─────────────────────────────────────────────────────────


def test_radiated_pressure_shape(km_eom, pulse, units, save_spec):
    result = run_simulation(km_eom, pulse, units=units, save_spec=save_spec)
    p_rad = result.radiated_pressure(d=1e-3)
    assert p_rad.shape == result.ts.shape


def test_radiated_pressure_is_finite(km_eom, pulse, units, save_spec):
    result = run_simulation(km_eom, pulse, units=units, save_spec=save_spec)
    p_rad = result.radiated_pressure(d=1e-3)
    assert bool(jnp.all(jnp.isfinite(p_rad)))


# ── compute_radius_metrics ────────────────────────────────────────────────────


def test_compute_radius_metrics_has_expected_keys(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    metrics = compute_radius_metrics(result)
    for key in ("max_radius", "min_radius", "max_ratio", "min_ratio", "swing_ratio"):
        assert key in metrics


def test_compute_radius_metrics_positive_values(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    metrics = compute_radius_metrics(result)
    for key, val in metrics.items():
        assert val > 0.0, f"{key} should be positive"


def test_compute_radius_metrics_max_gte_min(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    metrics = compute_radius_metrics(result)
    assert metrics["max_radius"] >= metrics["min_radius"]


# ── arrays_from_result ────────────────────────────────────────────────────────


def test_arrays_from_result_returns_numpy(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    arrays = arrays_from_result(result)
    assert isinstance(arrays.time_us, np.ndarray)
    assert isinstance(arrays.radius_um, np.ndarray)
    assert isinstance(arrays.pressure_kpa, np.ndarray)


def test_arrays_from_result_shapes(rp_eom, pulse, units, save_spec):
    result = run_simulation(rp_eom, pulse, units=units, save_spec=save_spec)
    arrays = arrays_from_result(result)
    n = save_spec.num_samples
    assert arrays.time_us.shape == (n,)
    assert arrays.radius_um.shape == (n,)
    assert arrays.pressure_kpa.shape == (n,)
