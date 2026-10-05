"""Neural-network-parameterised pulse for trainable waveform design."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp

from .base import Pulse

__all__ = ["NeuralPulse"]


class NeuralPulse(Pulse):
    r"""Acoustic pulse parameterised by a neural network.

    The network $f_\theta$ maps the normalised time since the pulse starts
    to the instantaneous pressure, scaled by `pressure_scale`:

    $$
    p(t) = P\, f_\theta\!\left(\frac{t - t_0}{T}\right) w(t - t_0, T)
    $$

    where $P$ is `pressure_scale`, $t_0$ is `initial_time`, $T$ is
    `pulse_duration`, and $w$ is `envelope`. The network input runs from 0
    to 1 across the active window, wherever the pulse starts.

    The network weights are the trainable state. Because the entire pulse
    is an Equinox module, they take part in JAX transformations (`jit`,
    `grad`, `vmap`), so you can optimise the driving waveform with
    gradient descent. `pulse_duration` and `pressure_scale` are static
    configuration: the constructor converts them to Python floats, and
    they aren't PyTree leaves, so an optimiser never updates them. A new
    value of either one makes `jax.jit` trace the pulse again.

    `NeuralPulse` inherits the `initial_time` and `envelope` fields from
    [`Pulse`][jbubble.pulse.base.Pulse]; set them as keyword arguments.

    Parameters
    ----------
    net : eqx.Module
        Any callable Equinox module, such as `eqx.nn.MLP`, that maps a
        1-element array to a scalar or 1-element output.
    pulse_duration : float
        Nominal pulse duration [s], used for time normalisation and the
        `duration` property. Static: it must be concrete, not traced.
    pressure_scale : float
        Multiplicative scaling applied to the network output [Pa]. Static:
        it must be concrete, not traced. Default: `1.0`.

    Examples
    --------
    >>> import jax
    >>> import equinox as eqx
    >>> from jbubble.pulse import NeuralPulse
    >>> key = jax.random.PRNGKey(0)
    >>> mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=32,
    ...                   depth=2, key=key)
    >>> pulse = NeuralPulse(net=mlp, pulse_duration=10e-6,
    ...                     pressure_scale=200e3)
    """

    net: eqx.Module
    pulse_duration: float = eqx.field(static=True, converter=float)
    pressure_scale: float = eqx.field(default=1.0, static=True, converter=float)

    @property
    def duration(self) -> float:
        """Nominal pulse duration, `pulse_duration` [s]."""
        return self.pulse_duration

    def _evaluate(self, t: jax.Array) -> jax.Array:
        t_norm = (t - self.initial_time) / self.pulse_duration
        net_fn = cast(Callable[[jax.Array], jax.Array], self.net)
        return net_fn(jnp.atleast_1d(t_norm)).squeeze() * self.pressure_scale
