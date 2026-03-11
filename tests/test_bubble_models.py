"""Tests for all concrete EoM compositions."""

import jax
import jax.numpy as jnp
import pytest
from jbubble.bubble import ConfinedBubbleState, SphericalConfinement

from conftest import ALL_EOM_FACTORIES, make_confinement

_R0 = 3e-6


# ── initial_state ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_initial_state_has_R_field(factory):
    eom = factory(_R0)
    s = eom.initial_state()
    assert hasattr(s, "R")
    assert hasattr(s, "R_dot")


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_initial_state_type(factory):
    eom = factory(_R0)
    s = eom.initial_state()
    if isinstance(eom, SphericalConfinement):
        assert isinstance(s, ConfinedBubbleState)
    else:
        from jbubble.bubble import BubbleState

        assert isinstance(s, BubbleState)


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_initial_state_R_is_R0(factory):
    eom = factory(_R0)
    s = eom.initial_state()
    assert float(s.R) == pytest.approx(_R0)


# ── __call__ (ODE RHS) ──────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_call_output_is_same_type(factory, pulse):
    eom = factory(_R0)
    s0 = eom.initial_state()
    dsdt = eom(jnp.array(0.0), s0, pulse)
    assert type(dsdt) is type(s0)


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_call_is_finite(factory, pulse):
    eom = factory(_R0)
    s0 = eom.initial_state()
    dsdt = eom(jnp.array(0.0), s0, pulse)
    leaves = jax.tree.leaves(dsdt)
    assert all(bool(jnp.isfinite(leaf)) for leaf in leaves)


@pytest.mark.parametrize("factory", ALL_EOM_FACTORIES)
def test_call_R_dot_is_velocity(factory, pulse):
    # d(R)/dt = Rdot; at initial state Rdot=0, so dsdt.R should be 0
    eom = factory(_R0)
    s0 = eom.initial_state()
    dsdt = eom(jnp.array(0.0), s0, pulse)
    assert float(dsdt.R) == pytest.approx(0.0, abs=1e-10)


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
def test_rescale_state_same_type(factory, units):
    eom = factory(_R0)
    s0 = eom.initial_state()
    rescaled = eom.rescale_state(s0, units)
    assert type(rescaled) is type(s0)


def test_confinement_rescale_state_4dof(units):
    """SphericalConfinement should rescale all four state components."""
    eom = make_confinement(_R0)
    ones = ConfinedBubbleState(
        R=jnp.array(1.0),
        R_dot=jnp.array(1.0),
        a=jnp.array(1.0),
        a_dot=jnp.array(1.0),
    )
    rescaled = eom.rescale_state(ones, units)
    assert isinstance(rescaled, ConfinedBubbleState)
    # R and a should get L_scale, Rdot and a_dot should get vel_scale
    assert float(rescaled.R) == pytest.approx(units.L_scale)
    assert float(rescaled.R_dot) == pytest.approx(units.vel_scale)
    assert float(rescaled.a) == pytest.approx(units.L_scale)
    assert float(rescaled.a_dot) == pytest.approx(units.vel_scale)
