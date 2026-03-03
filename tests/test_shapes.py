"""Tests for jbubble.shapes — pulse waveform library."""

import jax.numpy as jnp
import pytest
from jbubble.shapes import (
    Asymmetrical,
    NegativeQuadratic,
    Quadratic,
    Sawtooth,
    Sine,
    SlantedSine,
    Square,
    TimeDomainSawtooth,
    TimeDomainSquare,
    TimeDomainTriangle,
    Triangle,
)

# ── parametrised fixtures ────────────────────────────────────────────────────

SHAPE_CLASSES = [
    Sine,
    Sawtooth,
    Triangle,
    Quadratic,
    NegativeQuadratic,
    Asymmetrical,
    SlantedSine,
    Square,
    TimeDomainSquare,
    TimeDomainSawtooth,
    TimeDomainTriangle,
]

_FREQ = 300e3
_T_MID = 1.0 / _FREQ  # one full period in


@pytest.mark.parametrize("ShapeClass", SHAPE_CLASSES)
def test_shape_returns_scalar(ShapeClass):
    shape = ShapeClass()
    val = shape(jnp.array(_T_MID), freq=_FREQ, phase=0.0, initial_time=0.0)
    assert val.ndim == 0


@pytest.mark.parametrize("ShapeClass", SHAPE_CLASSES)
def test_shape_is_finite(ShapeClass):
    shape = ShapeClass()
    val = shape(jnp.array(_T_MID), freq=_FREQ, phase=0.0, initial_time=0.0)
    assert bool(jnp.isfinite(val))


@pytest.mark.parametrize("ShapeClass", SHAPE_CLASSES)
def test_shape_output_in_reasonable_range(ShapeClass):
    # All shapes should produce values bounded in [-2, 2]
    shape = ShapeClass()
    ts = jnp.linspace(0.0, 3.0 / _FREQ, 32)
    vals = jnp.array([shape(t, freq=_FREQ, phase=0.0, initial_time=0.0) for t in ts])
    assert bool(jnp.all(jnp.abs(vals) <= 2.0))


def test_sine_quarter_period_value():
    shape = Sine()
    # At t = T/4, sin(2π·f·T/4) = sin(π/2) = 1
    t_quarter = jnp.array(1.0 / (4.0 * _FREQ))
    val = shape(t_quarter, freq=_FREQ, phase=0.0, initial_time=0.0)
    assert float(val) == pytest.approx(1.0, abs=1e-5)


def test_sine_half_period_near_zero():
    shape = Sine()
    # At t = T/2, sin(π) ≈ 0
    t_half = jnp.array(1.0 / (2.0 * _FREQ))
    val = shape(t_half, freq=_FREQ, phase=0.0, initial_time=0.0)
    assert float(val) == pytest.approx(0.0, abs=1e-5)
