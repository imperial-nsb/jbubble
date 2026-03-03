"""Tests for all concrete bubble model implementations."""

import jax.numpy as jnp
import pytest
from jbubble.bubble import (
    ChurchGompertz,
    KellerMiksisGompertz,
    KelvinVoigtGompertz,
    LeightonGompertz,
    Marmottant,
    MarmottantGompertz,
    NeoHookeanGompertz,
    RayleighPlesset,
    SphericalConfinement,
)

ALL_BUBBLE_CLASSES = [
    RayleighPlesset,
    Marmottant,
    MarmottantGompertz,
    KellerMiksisGompertz,
    KelvinVoigtGompertz,
    NeoHookeanGompertz,
    ChurchGompertz,
    LeightonGompertz,
    SphericalConfinement,
]

_R0 = 3e-6


def _make_bubble(cls):
    return cls(R0=_R0)


# ── initial_state ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_initial_state_is_1d(BubbleClass):
    b = _make_bubble(BubbleClass)
    s = b.initial_state()
    assert s.ndim == 1


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_initial_state_size(BubbleClass):
    b = _make_bubble(BubbleClass)
    s = b.initial_state()
    expected = 4 if isinstance(b, SphericalConfinement) else 2
    assert s.shape == (expected,)


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_initial_state_first_element_is_R0(BubbleClass):
    b = _make_bubble(BubbleClass)
    s = b.initial_state()
    # First state element should be the equilibrium radius
    assert float(s[0]) == pytest.approx(_R0)


# ── bubble_equation ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_bubble_equation_output_shape(BubbleClass, pulse):
    b = _make_bubble(BubbleClass)
    s0 = b.initial_state()
    dsdt = b.bubble_equation(jnp.array(0.0), s0, pulse)
    assert dsdt.shape == s0.shape


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_bubble_equation_is_finite(BubbleClass, pulse):
    b = _make_bubble(BubbleClass)
    s0 = b.initial_state()
    dsdt = b.bubble_equation(jnp.array(0.0), s0, pulse)
    assert bool(jnp.all(jnp.isfinite(dsdt)))


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_bubble_equation_first_element_is_velocity(BubbleClass, pulse):
    # d(R)/dt = Ṙ; at initial state Ṙ=0, so dsdt[0] should be 0
    b = _make_bubble(BubbleClass)
    s0 = b.initial_state()
    dsdt = b.bubble_equation(jnp.array(0.0), s0, pulse)
    # With zero initial velocity, dR/dt = 0
    assert float(dsdt[0]) == pytest.approx(0.0, abs=1e-10)


# ── get_scaled ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_get_scaled_returns_same_type(BubbleClass, units):
    b = _make_bubble(BubbleClass)
    b_scaled = b.get_scaled(units)
    assert type(b_scaled) is type(b)


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_get_scaled_r0_is_dimensionless(BubbleClass, units):
    b = _make_bubble(BubbleClass)
    b_scaled = b.get_scaled(units)
    assert float(b_scaled.R0) == pytest.approx(_R0 / units.L_scale)


# ── surface_tension ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_surface_tension_finite_at_R0(BubbleClass):
    b = _make_bubble(BubbleClass)
    sigma = b.surface_tension(jnp.array(float(b.R0)))
    assert bool(jnp.isfinite(sigma))


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_surface_tension_non_negative(BubbleClass):
    b = _make_bubble(BubbleClass)
    # Surface tension must be non-negative at equilibrium radius
    sigma = b.surface_tension(jnp.array(float(b.R0)))
    assert float(sigma) >= 0.0


# ── rho_L consistency ────────────────────────────────────────────────────────


@pytest.mark.parametrize("BubbleClass", ALL_BUBBLE_CLASSES)
def test_has_rho_L_attribute(BubbleClass):
    """All models must expose rho_L so SimulationResult.radiated_pressure works."""
    b = _make_bubble(BubbleClass)
    assert hasattr(b, "rho_L"), f"{BubbleClass.__name__} missing rho_L attribute"


# ── RayleighPlesset-specific ──────────────────────────────────────────────────


def test_rayleigh_plesset_chi_R_is_zero():
    """RP model has no shell — chi_R should return zero."""
    b = RayleighPlesset(R0=_R0)
    radii = jnp.array([2e-6, 3e-6, 4e-6])
    chi = b.chi_R(radii)
    assert bool(jnp.all(chi == 0.0))
