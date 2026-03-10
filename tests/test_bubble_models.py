"""Tests for all concrete EoM compositions."""

import jax.numpy as jnp
import pytest
from jbubble.bubble import SphericalConfinement

from conftest import ALL_EOM_FACTORIES, make_confinement

_R0 = 3e-6


# ── initial_state ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_initial_state_is_1d(factory):
    eom = factory(_R0)
    s = eom.initial_state()
    assert s.ndim == 1


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_initial_state_size(factory):
    eom = factory(_R0)
    s = eom.initial_state()
    expected = 4 if isinstance(eom, SphericalConfinement) else 2
    assert s.shape == (expected,)


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_initial_state_first_element_is_R0(factory):
    eom = factory(_R0)
    s = eom.initial_state()
    assert float(s[0]) == pytest.approx(_R0)


# ── __call__ (ODE RHS) ──────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_call_output_shape(factory, pulse):
    eom = factory(_R0)
    s0 = eom.initial_state()
    dsdt = eom(jnp.array(0.0), s0, pulse)
    assert dsdt.shape == s0.shape


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_call_is_finite(factory, pulse):
    eom = factory(_R0)
    s0 = eom.initial_state()
    dsdt = eom(jnp.array(0.0), s0, pulse)
    assert bool(jnp.all(jnp.isfinite(dsdt)))


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_call_first_element_is_velocity(factory, pulse):
    # d(R)/dt = Rdot; at initial state Rdot=0, so dsdt[0] should be 0
    eom = factory(_R0)
    s0 = eom.initial_state()
    dsdt = eom(jnp.array(0.0), s0, pulse)
    assert float(dsdt[0]) == pytest.approx(0.0, abs=1e-10)


# ── get_scaled ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_get_scaled_returns_same_type(factory, units):
    eom = factory(_R0)
    eom_scaled = eom.get_scaled(units)
    assert type(eom_scaled) is type(eom)


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_get_scaled_r0_is_dimensionless(factory, units):
    eom = factory(_R0)
    eom_scaled = eom.get_scaled(units)
    assert float(eom_scaled.R0) == pytest.approx(_R0 / units.L_scale)


# ── surface_tension ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_surface_tension_finite_at_R0(factory):
    eom = factory(_R0)
    sigma = eom.surface_tension(jnp.array(float(eom.R0)))
    assert bool(jnp.isfinite(sigma))


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_surface_tension_non_negative(factory):
    eom = factory(_R0)
    sigma = eom.surface_tension(jnp.array(float(eom.R0)))
    assert float(sigma) >= 0.0


# ── rho_L consistency ────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_has_rho_L_attribute(factory):
    """All EoMs must expose rho_L so SimulationResult.radiated_pressure works."""
    eom = factory(_R0)
    assert hasattr(eom, "rho_L"), f"{type(eom).__name__} missing rho_L attribute"


# ── rescale_state ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_rescale_state_shape(factory, units):
    eom = factory(_R0)
    s0 = eom.initial_state()
    ones = jnp.ones_like(s0)
    rescaled = eom.rescale_state(ones, units)
    assert rescaled.shape == s0.shape


def test_confinement_rescale_state_4dof(units):
    """SphericalConfinement should rescale all four state components."""
    eom = make_confinement(_R0)
    ones = jnp.ones(4)
    rescaled = eom.rescale_state(ones, units)
    assert rescaled.shape == (4,)
    # R and a should get L_scale, Rdot and a_dot should get vel_scale
    assert float(rescaled[0]) == pytest.approx(units.L_scale)
    assert float(rescaled[1]) == pytest.approx(units.vel_scale)
    assert float(rescaled[2]) == pytest.approx(units.L_scale)
    assert float(rescaled[3]) == pytest.approx(units.vel_scale)
