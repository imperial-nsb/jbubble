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
from jbubble import Pulse, SaveSpec, run_simulation
from jbubble.bubble import BubbleState, KellerMiksis, RayleighPlesset

from conftest import make_gompertz_lipid, make_rp


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


def test_jit_eom_call(jax_pulse):
    eom = make_gompertz_lipid(KellerMiksis, R0=4e-6, c_L=1481.0)
    s0 = eom.initial_state()

    @jax.jit
    def f(eom_model, state):
        return eom_model(jnp.array(0.0), state, jax_pulse)

    result = f(eom, s0)
    assert isinstance(result, BubbleState)
    assert bool(jnp.isfinite(result.R))
    assert bool(jnp.isfinite(result.R_dot))


def test_jit_run_simulation(jax_pulse, jax_save_spec):
    """run_simulation must be JIT-compilable — critical for performance."""
    eom = make_rp(R0=3e-6)

    jit_run = jax.jit(
        run_simulation,
        static_argnames=("save_spec",),
    )
    result = jit_run(eom, jax_pulse, save_spec=jax_save_spec)
    assert result.radius.shape == (jax_save_spec.num_samples,)
    assert bool(jnp.all(jnp.isfinite(result.radius)))


# ── vmap ──────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_vmap_over_R0(jax_pulse, jax_save_spec):
    """Batch simulation over multiple radii — core use case for parameter sweeps."""

    def sim_for_R0(R0):
        eom = make_rp(R0)
        return run_simulation(eom, jax_pulse, save_spec=jax_save_spec).radius

    radii = jnp.array([2e-6, 3e-6, 4e-6])
    all_radii = jax.vmap(sim_for_R0)(radii)
    assert all_radii.shape == (3, jax_save_spec.num_samples)
    assert bool(jnp.all(jnp.isfinite(all_radii)))


# ── grad ──────────────────────────────────────────────────────────────────────


def test_grad_surface_tension_gompertz_is_finite():
    """Gompertz surface tension must be differentiable at all radii."""
    eom = make_gompertz_lipid(KellerMiksis, R0=4e-6, c_L=1481.0)
    s0 = eom.initial_state()

    def sigma_at_R(R):
        s = BubbleState(R=R, R_dot=jnp.zeros(()), R0=s0.R0, P_gas0=s0.P_gas0)
        return eom.shell.sigma(s)

    grad_fn = jax.grad(sigma_at_R)
    for r_factor in (0.5, 0.99, 1.0, 1.01, 1.05, 1.5):
        r = jnp.array(r_factor * float(eom.R0))
        g = grad_fn(r)
        assert bool(jnp.isfinite(g)), f"Non-finite grad at r_factor={r_factor}"


def test_grad_through_eom_call_wrt_R0(jax_pulse):
    """Gradient of ODE RHS scalar w.r.t. R0 — exercises full pytree machinery."""

    def loss(r0: jax.Array) -> jax.Array:
        eom = make_gompertz_lipid(KellerMiksis, R0=r0, c_L=1481.0)
        s_perturbed = BubbleState(R=1.2 * r0, R_dot=jnp.zeros(()))
        dsdt = eom(jnp.array(0.0), s_perturbed, jax_pulse)
        return dsdt.R**2 + dsdt.R_dot**2

    r0_val = jnp.array(4e-6)
    grad_val = jax.grad(loss)(r0_val)
    assert bool(jnp.isfinite(grad_val)), f"Got non-finite grad: {grad_val}"
    assert grad_val != 0.0
