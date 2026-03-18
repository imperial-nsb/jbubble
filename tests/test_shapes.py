"""Tests for jbubble.pulse.shapes."""

import jax
import jax.numpy as jnp
import pytest
from jbubble.pulse.shapes import (
    InvertedSawtooth,
    NegativeQuadratic,
    Quadratic,
    Rectangular,
    Sawtooth,
    Sine,
    Square,
    TimeDomainSawtooth,
    TimeDomainSquare,
    TimeDomainTriangle,
    Triangle,
)

FREQ = 1e6
PHASE = 0.0
T0 = 0.0


class TestSine:
    def test_zero_at_t0(self):
        shape = Sine()
        val = float(shape(jnp.asarray(0.0), FREQ, PHASE, T0))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_one_at_quarter_period(self):
        shape = Sine()
        t = 0.25 / FREQ  # quarter period
        val = float(shape(jnp.asarray(t), FREQ, PHASE, T0))
        assert val == pytest.approx(1.0, abs=1e-10)

    def test_periodic(self):
        shape = Sine()
        t = jnp.asarray(1.7e-6)
        period = 1.0 / FREQ
        v1 = float(shape(t, FREQ, PHASE, T0))
        v2 = float(shape(t + period, FREQ, PHASE, T0))
        assert v1 == pytest.approx(v2, abs=1e-10)

    def test_differentiable(self):
        shape = Sine()
        t = jnp.asarray(0.3e-6)
        grad = jax.grad(lambda t: shape(t, FREQ, PHASE, T0))(t)
        assert jnp.isfinite(grad)

    def test_name(self):
        assert Sine().name == "sine"


class TestFourierShapes:
    """Common tests for all Fourier-series-based shapes."""

    @pytest.fixture(
        params=[Square, Sawtooth, InvertedSawtooth, Triangle, Quadratic],
        ids=["Square", "Sawtooth", "InvSawtooth", "Triangle", "Quadratic"],
    )
    def shape(self, request):
        return request.param()

    def test_bounded_output(self, shape):
        ts = jnp.linspace(0, 2.0 / FREQ, 500)
        vals = jax.vmap(lambda t: shape(t, FREQ, PHASE, T0))(ts)
        assert (
            float(jnp.max(jnp.abs(vals))) < 1.5
        )  # bounded (Gibbs overshoot up to ~9%)

    def test_finite_output(self, shape):
        ts = jnp.linspace(0, 1.0 / FREQ, 100)
        vals = jax.vmap(lambda t: shape(t, FREQ, PHASE, T0))(ts)
        assert jnp.all(jnp.isfinite(vals))

    def test_has_name(self, shape):
        assert isinstance(shape.name, str)
        assert len(shape.name) > 0


class TestTimeDomainShapes:
    @pytest.fixture(
        params=[TimeDomainSquare, TimeDomainSawtooth, TimeDomainTriangle],
        ids=["TDSquare", "TDSawtooth", "TDTriangle"],
    )
    def shape(self, request):
        return request.param()

    def test_bounded_output(self, shape):
        ts = jnp.linspace(0, 2.0 / FREQ, 500)
        vals = jax.vmap(lambda t: shape(t, FREQ, PHASE, T0))(ts)
        assert float(jnp.max(jnp.abs(vals))) <= 1.01

    def test_finite_output(self, shape):
        ts = jnp.linspace(0, 1.0 / FREQ, 100)
        vals = jax.vmap(lambda t: shape(t, FREQ, PHASE, T0))(ts)
        assert jnp.all(jnp.isfinite(vals))

    def test_differentiable(self, shape):
        t = jnp.asarray(0.3e-6)
        grad = jax.grad(lambda t: shape(t, FREQ, PHASE, T0))(t)
        assert jnp.isfinite(grad)


class TestNegativeQuadratic:
    def test_negates_quadratic(self):
        q = Quadratic()
        nq = NegativeQuadratic()
        t = jnp.asarray(0.3e-6)
        v_q = float(q(t, FREQ, PHASE, T0))
        v_nq = float(nq(t, FREQ, PHASE, T0))
        assert v_nq == pytest.approx(-v_q, rel=1e-10)


class TestRectangular:
    def test_duty_cycle(self):
        shape = Rectangular(duty=0.5)
        ts = jnp.linspace(0, 1.0 / FREQ, 500)
        vals = jax.vmap(lambda t: shape(t, FREQ, PHASE, T0))(ts)
        assert jnp.all(jnp.isfinite(vals))

    def test_name_includes_duty(self):
        shape = Rectangular(duty=0.25)
        assert "0.25" in shape.name

    def test_dc_offset(self):
        shape = Rectangular(duty=0.5, high_level=1.0, low_level=-1.0)
        # DC offset = 1.0 * 0.5 + (-1.0) * 0.5 = 0.0
        assert shape.dc_offset == pytest.approx(0.0, abs=1e-10)

    def test_asymmetric_dc(self):
        shape = Rectangular(duty=0.25, high_level=1.0, low_level=0.0)
        assert shape.dc_offset == pytest.approx(0.25, abs=1e-10)


class TestTimeDomainSquare:
    def test_sharpness(self):
        sharp = TimeDomainSquare(sharpness=100.0)
        soft = TimeDomainSquare(sharpness=5.0)
        # At quarter period (peak of sin), both should be near 1.0 but sharp more so
        t = jnp.asarray(0.25 / FREQ)
        v_sharp = float(sharp(t, FREQ, PHASE, T0))
        v_soft = float(soft(t, FREQ, PHASE, T0))
        assert abs(v_sharp) > abs(v_soft)
