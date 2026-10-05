"""Tests for jbubble.pulse — ToneBurst, ChirpPulse, SampledPulse, NeuralPulse, composition."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jbubble import SaveSpec, run_simulation
from jbubble.pulse import (
    ChirpPulse,
    HannEnvelope,
    NeuralPulse,
    Offset,
    SampledPulse,
    Scaled,
    Summed,
    ToneBurst,
)
from jbubble.pulse.chirp import ExponentialSweep
from jbubble.pulse.shapes import Sine
from jbubble.utils.presets import free_bubble


class TestToneBurst:
    def test_duration(self):
        pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)
        assert float(pulse.duration) == pytest.approx(5e-6, rel=1e-10)

    def test_zero_before_pulse(self):
        pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)
        assert float(pulse(jnp.asarray(-1e-6))) == pytest.approx(0.0, abs=1.0)

    def test_zero_after_pulse(self):
        pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)
        assert float(pulse(jnp.asarray(20e-6))) == pytest.approx(0.0, abs=1.0)

    def test_peak_amplitude_near_pressure(self):
        pulse = ToneBurst(freq=1e6, pressure=200e3, shape=Sine(), cycle_num=10)
        ts = jnp.linspace(0, 10e-6, 10000)
        ps = jax.vmap(pulse)(ts)
        peak = float(jnp.max(jnp.abs(ps)))
        assert peak == pytest.approx(200e3, rel=0.05)

    def test_differentiable(self):
        pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)
        t = jnp.asarray(2.5e-6)
        dp_dt = jax.grad(pulse)(t)
        assert jnp.isfinite(dp_dt)

    def test_t_end(self):
        pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)
        # t_end = initial_time + 2 * duration = 0 + 2 * 5e-6 = 10e-6
        assert float(pulse.t_end) == pytest.approx(10e-6, rel=1e-10)

    def test_with_phase_offset(self):
        pulse = ToneBurst(
            freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5, phase=jnp.pi / 2
        )
        t = jnp.asarray(2.5e-6)
        assert jnp.isfinite(pulse(t))


class TestChirpPulse:
    def test_creates_with_linear_sweep(self):
        pulse = ChirpPulse(
            freq_start=0.5e6,
            freq_end=2e6,
            pressure=100e3,
            sweep_duration=10e-6,
        )
        assert float(pulse.duration) == pytest.approx(10e-6, rel=1e-10)

    def test_creates_with_exponential_sweep(self):
        pulse = ChirpPulse(
            freq_start=0.5e6,
            freq_end=2e6,
            pressure=100e3,
            sweep_duration=10e-6,
            sweep=ExponentialSweep(),
        )
        t = jnp.asarray(5e-6)
        assert jnp.isfinite(pulse(t))

    def test_evaluates_within_sweep(self):
        pulse = ChirpPulse(
            freq_start=0.5e6,
            freq_end=2e6,
            pressure=100e3,
            sweep_duration=10e-6,
        )
        ts = jnp.linspace(0, 10e-6, 1000)
        ps = jax.vmap(pulse)(ts)
        assert jnp.all(jnp.isfinite(ps))
        assert float(jnp.max(jnp.abs(ps))) > 0

    def test_differentiable(self):
        pulse = ChirpPulse(
            freq_start=0.5e6,
            freq_end=2e6,
            pressure=100e3,
            sweep_duration=10e-6,
        )
        t = jnp.asarray(5e-6)
        dp_dt = jax.grad(pulse)(t)
        assert jnp.isfinite(dp_dt)


class TestSampledPulse:
    def test_interpolation(self):
        ts = jnp.linspace(0, 10e-6, 1000)
        ps = 200e3 * jnp.sin(2 * jnp.pi * 1e6 * ts)
        pulse = SampledPulse(ts=ts, pressures=ps)
        # Pick t = 5.25 µs (mid-pulse, envelope ≈ 1)
        # sin(2π*1e6*5.25e-6) = sin(10.5π) = sin(0.5π) = 1.0
        t_query = jnp.asarray(5.25e-6)
        result = float(pulse(t_query))
        assert result == pytest.approx(200e3, rel=0.01)

    def test_duration(self):
        ts = jnp.linspace(0, 10e-6, 100)
        ps = jnp.zeros(100)
        pulse = SampledPulse(ts=ts, pressures=ps)
        assert pulse.duration == pytest.approx(10e-6, rel=1e-10)

    def test_from_uniform(self):
        ps = jnp.ones(100)
        pulse = SampledPulse.from_uniform(ps, dt=1e-7)
        assert pulse.duration == pytest.approx(99 * 1e-7, rel=1e-6)

    def test_differentiable(self):
        ts = jnp.linspace(0, 10e-6, 100)
        ps = 100e3 * jnp.sin(2 * jnp.pi * 1e6 * ts)
        pulse = SampledPulse(ts=ts, pressures=ps)
        t = jnp.asarray(3e-6)
        dp_dt = jax.grad(pulse)(t)
        assert jnp.isfinite(dp_dt)


class TestNeuralPulse:
    def test_creates_and_evaluates(self):
        key = jax.random.PRNGKey(0)
        mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
        pulse = NeuralPulse(net=mlp, pulse_duration=10e-6, pressure_scale=100e3)
        t = jnp.asarray(5e-6)
        result = pulse(t)
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_duration(self):
        key = jax.random.PRNGKey(0)
        mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
        pulse = NeuralPulse(net=mlp, pulse_duration=10e-6)
        assert float(pulse.duration) == pytest.approx(10e-6, rel=1e-10)

    def test_differentiable(self):
        key = jax.random.PRNGKey(0)
        mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
        pulse = NeuralPulse(net=mlp, pulse_duration=10e-6, pressure_scale=100e3)
        t = jnp.asarray(5e-6)
        dp_dt = jax.grad(pulse)(t)
        assert jnp.isfinite(dp_dt)


class TestPulseComposition:
    @pytest.fixture
    def p1(self):
        return ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)

    @pytest.fixture
    def p2(self):
        return ToneBurst(freq=2e6, pressure=50e3, shape=Sine(), cycle_num=10)

    def test_add_pulses(self, p1, p2):
        combined = p1 + p2
        assert isinstance(combined, Summed)
        t = jnp.asarray(2e-6)
        assert jnp.isfinite(combined(t))

    def test_add_float(self, p1):
        offset_pulse = p1 + 1000.0
        assert isinstance(offset_pulse, Offset)
        t = jnp.asarray(2e-6)
        base_val = float(p1(t))
        offset_val = float(offset_pulse(t))
        assert offset_val == pytest.approx(base_val + 1000.0, rel=1e-6)

    def test_radd_float(self, p1):
        offset_pulse = 1000.0 + p1
        assert isinstance(offset_pulse, Offset)

    def test_mul_float(self, p1):
        scaled = p1 * 2.0
        assert isinstance(scaled, Scaled)
        t = jnp.asarray(2e-6)
        assert float(scaled(t)) == pytest.approx(2.0 * float(p1(t)), rel=1e-6)

    def test_rmul_float(self, p1):
        scaled = 0.5 * p1
        assert isinstance(scaled, Scaled)
        t = jnp.asarray(2e-6)
        assert float(scaled(t)) == pytest.approx(0.5 * float(p1(t)), rel=1e-6)

    def test_neg(self, p1):
        neg = -p1
        assert isinstance(neg, Scaled)
        t = jnp.asarray(2e-6)
        assert float(neg(t)) == pytest.approx(-float(p1(t)), rel=1e-6)

    def test_sub_pulse(self, p1, p2):
        diff = p1 - p2
        t = jnp.asarray(2e-6)
        expected = float(p1(t)) - float(p2(t))
        assert float(diff(t)) == pytest.approx(expected, rel=1e-6)

    def test_sub_float(self, p1):
        shifted = p1 - 500.0
        assert isinstance(shifted, Offset)
        t = jnp.asarray(2e-6)
        assert float(shifted(t)) == pytest.approx(float(p1(t)) - 500.0, rel=1e-6)

    def test_div(self, p1):
        halved = p1 / 2.0
        assert isinstance(halved, Scaled)
        t = jnp.asarray(2e-6)
        assert float(halved(t)) == pytest.approx(float(p1(t)) / 2.0, rel=1e-6)

    def test_pos(self, p1):
        same = +p1
        assert same is p1

    def test_windowed(self, p1):
        windowed = p1.windowed(HannEnvelope())
        t = jnp.asarray(2.5e-6)  # middle of burst
        assert jnp.isfinite(windowed(t))

    def test_summed_t_end(self, p1, p2):
        combined = p1 + p2
        # t_end should be the max of both children's t_end
        assert float(combined.t_end) == max(float(p1.t_end), float(p2.t_end))

    def test_summed_flat(self, p1, p2):
        """Adding already-summed pulses should flatten."""
        s1 = p1 + p2
        p3 = ToneBurst(freq=3e6, pressure=30e3, shape=Sine(), cycle_num=15)
        s2 = s1 + p3
        assert isinstance(s2, Summed)
        assert len(s2.pulses) == 3


# Near the pulse start the Hann window is about 0.02 while the default
# soft-rectangular envelope is about 1, so windowing is clearly visible.
T_EDGE = jnp.asarray(0.25e-6)
T_MID = jnp.asarray(2.5e-6)


@pytest.fixture
def tone_early():
    return ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)


@pytest.fixture
def tone_late():
    return ToneBurst(
        freq=2.5e6, pressure=30e3, shape=Sine(), cycle_num=15, initial_time=6e-6
    )


class TestCompositeWindowed:
    """``windowed`` on Scaled and Offset reaches the child pulse."""

    def test_scaled_windowed_applies_envelope(self, tone_early):
        got = (tone_early * 2.0).windowed(HannEnvelope())(T_EDGE)
        want = 2.0 * tone_early.windowed(HannEnvelope())(T_EDGE)
        assert float(got) == pytest.approx(float(want), rel=1e-9, abs=1e-9)

    def test_offset_windowed_applies_envelope_to_child(self, tone_early):
        got = (tone_early + 1000.0).windowed(HannEnvelope())(T_EDGE)
        want = tone_early.windowed(HannEnvelope())(T_EDGE) + 1000.0
        assert float(got) == pytest.approx(float(want), rel=1e-9, abs=1e-9)

    def test_offset_constant_stays_unwindowed(self, tone_early):
        windowed = (tone_early + 1000.0).windowed(HannEnvelope())
        assert float(windowed(jnp.asarray(-1e-6))) == pytest.approx(1000.0)

    def test_scaled_sum_is_windowed(self, tone_early, tone_late):
        final = ((tone_early + tone_late) * 0.7).windowed(HannEnvelope())
        want = 0.7 * (tone_early + tone_late).windowed(HannEnvelope())(T_EDGE)
        assert float(final(T_EDGE)) == pytest.approx(float(want), rel=1e-9, abs=1e-9)
        unwindowed = 0.7 * (tone_early + tone_late)(T_EDGE)
        assert abs(float(final(T_EDGE))) < 0.2 * abs(float(unwindowed))

    def test_windowed_keeps_type(self, tone_early):
        assert isinstance((tone_early * 2.0).windowed(HannEnvelope()), Scaled)
        assert isinstance((tone_early + 1.0).windowed(HannEnvelope()), Offset)


class TestSummedUnderTracing:
    """A Summed pulse works under jit, so it can be simulated."""

    def test_duration_under_jit(self, tone_early, tone_late):
        duration = jax.jit(lambda s: s.duration)(tone_early + tone_late)
        assert float(duration) == pytest.approx(12e-6, rel=1e-12)

    def test_t_end_under_jit(self, tone_early, tone_late):
        t_end = jax.jit(lambda s: s.t_end)(tone_early + tone_late)
        assert float(t_end) == pytest.approx(18e-6, rel=1e-12)

    def test_matches_python_max(self, tone_early, tone_late):
        summed = tone_early + tone_late
        assert float(summed.t_end) == max(
            float(tone_early.t_end), float(tone_late.t_end)
        )

    def test_run_simulation_with_summed_pulse(self, tone_early, tone_late):
        eom, _ = free_bubble()
        result = jax.jit(run_simulation)(
            eom, (tone_early + tone_late) * 0.5, save_spec=SaveSpec(num_samples=64)
        )
        assert bool(result.converged)
        assert float(result.ts[-1]) == pytest.approx(18e-6, rel=1e-9)

    def test_grad_through_summed(self, tone_early):
        def value(pressure):
            other = ToneBurst(freq=2e6, pressure=pressure, shape=Sine(), cycle_num=5)
            return (tone_early + other)(T_MID)

        assert jnp.isfinite(jax.grad(value)(jnp.asarray(10e3)))
