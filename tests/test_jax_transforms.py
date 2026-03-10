"""Tests for JAX transforms (jit, vmap, grad) with jbubble objects.

These are the most important tests for an autodiff-first library.
They verify that:
  - Equinox modules are valid JAX pytrees (jit/vmap/grad compatible)
  - Gompertz surface tension is smooth and differentiable everywhere
  - ODE simulation is JIT-compilable
  - vmap over bubble parameters works correctly
"""

import jax
import jax.numpy as jnp
import jbubble.shapes as shapes
import pytest
from jbubble import Pulse, SaveSpec, Units, run_simulation
from jbubble.bubble import KellerMiksis, RayleighPlesset

from conftest import make_gompertz_lipid, make_rp

# ── session fixtures (compile once per test session) ──────────────────────────


@pytest.fixture(scope="module")
def jax_units():
    return Units()


@pytest.fixture(scope="module")
def jax_pulse():
    return Pulse(
        freq=300e3,
        pressure=50e3,
        shape=shapes.Sine(),
        cycle_num=4,
        initial_time=1e-6,
        apply_hann=False,
    )


@pytest.fixture(scope="module")
def jax_save_spec():
    return SaveSpec(num_samples=32)


# ── jit ───────────────────────────────────────────────────────────────────────


def test_jit_eom_call(jax_units, jax_pulse):
    eom = make_gompertz_lipid(KellerMiksis, R0=4e-6, c_L=1481.0)
    scaled_eom = eom.get_scaled(jax_units)
    scaled_p = jax_pulse.get_scaled(jax_units)
    s0 = scaled_eom.initial_state()

    @jax.jit
    def f(eom_model, state):
        return eom_model(jnp.array(0.0), state, scaled_p)

    result = f(scaled_eom, s0)
    assert result.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_jit_run_simulation(jax_units, jax_pulse, jax_save_spec):
    """run_simulation must be JIT-compilable — critical for performance."""
    eom = make_rp(R0=3e-6)

    jit_run = jax.jit(
        run_simulation,
        static_argnames=("save_spec",),
    )
    result = jit_run(eom, jax_pulse, units=jax_units, save_spec=jax_save_spec)
    assert result.radius.shape == (jax_save_spec.num_samples,)
    assert bool(jnp.all(jnp.isfinite(result.radius)))


# ── vmap ──────────────────────────────────────────────────────────────────────


def test_vmap_surface_tension(jax_units):
    eom = make_gompertz_lipid(KellerMiksis, R0=4e-6, c_L=1481.0).get_scaled(jax_units)
    radii = jnp.linspace(0.5 * float(eom.R0), 2.0 * float(eom.R0), 8)
    sigmas = jax.vmap(eom.surface_tension)(radii)
    assert sigmas.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(sigmas)))


@pytest.mark.slow
def test_vmap_over_R0(jax_units, jax_pulse, jax_save_spec):
    """Batch simulation over multiple radii — core use case for parameter sweeps."""

    def sim_for_R0(R0):
        eom = make_rp(R0)
        return run_simulation(
            eom, jax_pulse, units=jax_units, save_spec=jax_save_spec
        ).radius

    radii = jnp.array([2e-6, 3e-6, 4e-6])
    all_radii = jax.vmap(sim_for_R0)(radii)
    assert all_radii.shape == (3, jax_save_spec.num_samples)
    assert bool(jnp.all(jnp.isfinite(all_radii)))


# ── grad ──────────────────────────────────────────────────────────────────────


def test_grad_surface_tension_gompertz_is_finite(jax_units):
    """Gompertz surface tension must be differentiable at all radii."""
    eom = make_gompertz_lipid(KellerMiksis, R0=4e-6, c_L=1481.0).get_scaled(jax_units)
    grad_fn = jax.grad(eom.surface_tension)
    # Test across the buckled, elastic, and ruptured regimes
    for r_factor in (0.5, 0.99, 1.0, 1.01, 1.05, 1.5):
        r = jnp.array(r_factor * float(eom.R0))
        g = grad_fn(r)
        assert bool(jnp.isfinite(g)), f"Non-finite grad at r_factor={r_factor}"


def test_grad_through_eom_call_wrt_R0(jax_units, jax_pulse):
    """Gradient of ODE RHS scalar w.r.t. R0 — exercises full pytree machinery.

    Differentiate a scalar summary of the scaled EoM __call__ w.r.t. R0
    (in physical units). We evaluate at a perturbed radius so forces != 0.
    """
    scaled_p = jax_pulse.get_scaled(jax_units)

    def loss(r0_physical: jax.Array) -> jax.Array:
        eom = make_gompertz_lipid(KellerMiksis, R0=r0_physical, c_L=1481.0)
        scaled_eom = eom.get_scaled(jax_units)
        # Perturbed dimensionless state: R = 1.2 * R0_dim, Rdot = 0
        r0_dim = r0_physical / jax_units.L_scale
        s_perturbed = jnp.stack([1.2 * r0_dim, jnp.zeros(())])
        dsdt = scaled_eom(jnp.array(0.0), s_perturbed, scaled_p)
        return jnp.sum(dsdt**2)

    r0_val = jnp.array(4e-6)
    grad_val = jax.grad(loss)(r0_val)
    assert bool(jnp.isfinite(grad_val)), f"Got non-finite grad: {grad_val}"
    assert grad_val != 0.0  # gradient should be non-trivial at perturbed state
