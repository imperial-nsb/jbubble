"""Driving pulse parameterisations."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .units import Units
from .shapes import PulseShape


class Pulse(eqx.Module):
    """Pulse envelope shared between differentiable and interactive runs."""

    freq: float
    pressure: float
    shape: PulseShape
    phase: float = 0.0
    initial_time: float = 0.0
    cycle_num: float = 4.0
    apply_hann: bool = eqx.field(default=False, static=True)

    def __call__(self, t: jax.Array) -> jax.Array:
        pulse_span = self.cycle_num / self.freq
        tau = t - self.initial_time
        in_pulse = (tau >= 0) & (tau <= pulse_span)
        if self.apply_hann:
            hann = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau / pulse_span))
            window = jnp.where(in_pulse, hann, 0.0)
        else:
            window = jnp.where(in_pulse, 1.0, 0.0)

        val = self.shape(t, self.freq, self.phase, self.initial_time)
        return val * self.pressure * window

    def get_scaled(self, units: Units) -> "Pulse":
        return Pulse(
            freq=self.freq / units.freq_scale,
            pressure=self.pressure / units.P_scale,
            shape=self.shape,
            phase=self.phase,
            initial_time=self.initial_time / units.T_scale,
            cycle_num=self.cycle_num,
            apply_hann=self.apply_hann,
        )
