"""Tests for jbubble.solver — ODE solver wrapper."""

import jax.numpy as jnp
from jbubble import SaveSpec
from jbubble.solver import solve_eom


def test_solve_eom_output_ts_shape(rp_eom, pulse, save_spec):
    sol = solve_eom(rp_eom, pulse, t_span=(0.0, 20e-6), save_spec=save_spec)
    assert sol.ts is not None
    assert sol.ts.shape == (save_spec.num_samples,)


def test_solve_eom_output_ys_R_shape(rp_eom, pulse, save_spec):
    sol = solve_eom(rp_eom, pulse, t_span=(0.0, 20e-6), save_spec=save_spec)
    assert sol.ys is not None
    assert sol.ys.R.shape == (save_spec.num_samples,)
    assert sol.ys.R_dot.shape == (save_spec.num_samples,)


def test_solve_eom_radius_positive(rp_eom, pulse, save_spec):
    """Radius must stay positive for a stable (low-amplitude) simulation."""
    sol = solve_eom(rp_eom, pulse, t_span=(0.0, 20e-6), save_spec=save_spec)
    assert sol.ys is not None
    assert bool(jnp.all(sol.ys.R > 0.0))


def test_solve_eom_default_t_span_from_pulse(rp_eom, pulse):
    """When t_span is None it should be inferred from pulse duration."""
    ss = SaveSpec(num_samples=32)
    sol = solve_eom(rp_eom, pulse, save_spec=ss)
    assert sol.ts is not None
    assert sol.ts.shape == (32,)


def test_solve_eom_converges(rp_eom, pulse, save_spec):
    """diffrax should signal successful integration for benign parameters."""
    import diffrax

    sol = solve_eom(rp_eom, pulse, t_span=(0.0, 20e-6), save_spec=save_spec)
    assert bool(diffrax.is_successful(sol.result))
