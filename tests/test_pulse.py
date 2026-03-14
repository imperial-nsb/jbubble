"""Tests for jbubble.pulse — driving pulse parameterisation."""

import jax.numpy as jnp
import jbubble.shapes as shapes
import pytest
from jbubble import Pulse


def test_pulse_zero_before_initial_time(pulse):
    # t = 0 < initial_time = 1e-6 → pulse not yet started
    val = pulse(jnp.array(0.0))
    assert float(val) == pytest.approx(0.0)


def test_pulse_nonzero_during_burst(pulse):
    # t = initial_time + T/2 is inside the burst
    t_mid = jnp.array(1e-6 + 1.0 / (2.0 * 300e3))
    val = pulse(t_mid)
    assert float(jnp.abs(val)) > 0.0


def test_pulse_zero_after_burst(pulse):
    # t = initial_time + 20 cycles >> 4-cycle burst
    t_after = jnp.array(1e-6 + 20.0 / 300e3)
    val = pulse(t_after)
    assert float(val) == pytest.approx(0.0)


def test_pulse_amplitude_bounded_by_pressure(pulse):
    # |pulse(t)| ≤ pressure everywhere
    ts = jnp.linspace(0.0, 30e-6, 256)
    vals = jnp.abs(jnp.array([pulse(t) for t in ts]))
    assert bool(jnp.all(vals <= pulse.pressure + 1.0))  # +1 Pa tolerance


def test_pulse_hann_differs_from_no_hann():
    p_hann = Pulse(
        freq=300e3,
        pressure=50e3,
        shape=shapes.Sine(),
        cycle_num=4,
        initial_time=0.0,
        apply_hann=True,
    )
    p_rect = Pulse(
        freq=300e3,
        pressure=50e3,
        shape=shapes.Sine(),
        cycle_num=4,
        initial_time=0.0,
        apply_hann=False,
    )
    # At the beginning of the pulse the Hann window tapers to zero
    # while the rectangular window does not
    t_start = jnp.array(1e-9)  # just inside the pulse
    assert float(jnp.abs(p_hann(t_start))) < float(jnp.abs(p_rect(t_start)))
