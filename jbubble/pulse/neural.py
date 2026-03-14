"""Neural-network-parameterised pulse for trainable waveform design."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .base import Pulse


class NeuralPulse(Pulse):
    """Acoustic pulse parameterised by a neural network.

    The network maps normalised time ``t / duration`` to instantaneous
    pressure, scaled by ``pressure_scale``.  Because the entire pulse is an
    Equinox module, its parameters participate in JAX transformations
    (``jit``, ``grad``, ``vmap``) — making it straightforward to optimise
    the driving waveform via gradient descent.

    Parameters
    ----------
    net : eqx.Module
        Any callable Equinox module (e.g. ``eqx.nn.MLP``) mapping a
        scalar (or 1-element array) to a scalar output.
    pulse_duration : float
        Nominal pulse duration [s] — used for time normalisation and
        the :attr:`duration` property.
    pressure_scale : float
        Multiplicative scaling applied to the network output [Pa].
        Default: 1.0.

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
    pulse_duration: float
    pressure_scale: float = 1.0

    @property
    def duration(self) -> float:
        return self.pulse_duration

    def __call__(self, t: jax.Array) -> jax.Array:
        t_norm = t / self.pulse_duration
        return self.net(jnp.atleast_1d(t_norm)).squeeze() * self.pressure_scale
