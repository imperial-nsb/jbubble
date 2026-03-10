"""Tests for jbubble.solver — ODE solver wrapper."""

import jax.numpy as jnp
from jbubble import SaveSpec
from jbubble.solver import solve_eom


def test_solve_eom_output_ts_shape(rp_eom, pulse, units, save_spec):
    scaled_eom = rp_eom.get_scaled(units)
    scaled_p = pulse.get_scaled(units)
    t_span = (0.0, 20e-6 / units.T_scale)
    sol = solve_eom(scaled_eom, scaled_p, t_span=t_span, save_spec=save_spec)
    assert sol.ts is not None
    assert sol.ts.shape == (save_spec.num_samples,)


def test_solve_eom_output_ys_shape(rp_eom, pulse, units, save_spec):
    scaled_eom = rp_eom.get_scaled(units)
    scaled_p = pulse.get_scaled(units)
    t_span = (0.0, 20e-6 / units.T_scale)
    sol = solve_eom(scaled_eom, scaled_p, t_span=t_span, save_spec=save_spec)
    assert sol.ys is not None
    assert sol.ys.shape == (save_spec.num_samples, 2)


def test_solve_eom_radius_positive(rp_eom, pulse, units, save_spec):
    """Radius must stay positive for a stable (low-amplitude) simulation."""
    scaled_eom = rp_eom.get_scaled(units)
    scaled_p = pulse.get_scaled(units)
    t_span = (0.0, 20e-6 / units.T_scale)
    sol = solve_eom(scaled_eom, scaled_p, t_span=t_span, save_spec=save_spec)
    assert sol.ys is not None
    assert bool(jnp.all(sol.ys[:, 0] > 0.0))


def test_solve_eom_default_t_span_from_pulse(rp_eom, pulse, units):
    """When t_span is None it should be inferred from pulse duration."""
    scaled_eom = rp_eom.get_scaled(units)
    scaled_p = pulse.get_scaled(units)
    ss = SaveSpec(num_samples=32)
    sol = solve_eom(scaled_eom, scaled_p, save_spec=ss)
    assert sol.ts is not None
    assert sol.ts.shape == (32,)


def test_solve_eom_converges(rp_eom, pulse, units, save_spec):
    """diffrax should signal successful integration for benign parameters."""
    import diffrax

    scaled_eom = rp_eom.get_scaled(units)
    scaled_p = pulse.get_scaled(units)
    t_span = (0.0, 20e-6 / units.T_scale)
    sol = solve_eom(scaled_eom, scaled_p, t_span=t_span, save_spec=save_spec)
    assert bool(diffrax.is_successful(sol.result))
