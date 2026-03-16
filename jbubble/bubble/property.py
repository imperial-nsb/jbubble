"""State-dependent property abstractions for bubble models.

A ``Property`` is any callable ``eqx.Module`` that maps a ``BubbleState``
to a scalar array.  This abstraction covers:

- ``ConstantProperty`` — a fixed value (the common case)
- State-dependent laws like ``MarmottantSurfaceTension`` and
  ``GompertzSurfaceTension`` (defined in ``shell.py``)
- ``NeuralProperty`` — a neural network that learns an unknown law from data

Because all ``Property`` subclasses are Equinox modules, they are full JAX
pytrees: ``jit``, ``vmap``, and ``grad`` flow through them without any extra
bookkeeping.  The ``as_property`` converter lets model constructors accept
plain floats while still storing a proper ``Property`` internally.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .state import BubbleState


class Property(eqx.Module, abc.ABC):
    """Abstract base for state-dependent (or constant) bubble properties.

    Any callable ``eqx.Module`` that maps a ``BubbleState`` to a scalar
    array should inherit from this class.  The concrete subclass
    ``ConstantProperty`` handles the common case of a fixed value;
    state-dependent models (e.g. ``MarmottantSurfaceTension``) override
    ``__call__`` directly.
    """

    @abc.abstractmethod
    def __call__(self, state: BubbleState) -> jax.Array:
        """Evaluate the property at *state*.

        Parameters
        ----------
        state : BubbleState
            Current bubble state.

        Returns
        -------
        jax.Array
            Scalar value of the property.
        """
        ...


class ConstantProperty(Property):
    """A property that returns a constant value regardless of state.

    Fields
    ------
    val : float or jax.Array
        The constant value.  May be a JAX array (including a traced value
        inside ``jax.grad`` / ``jax.jit``) so that gradients flow through.
    """

    val: float | jax.Array

    def __call__(self, state: BubbleState) -> jax.Array:
        return self.val + state.R * 0.0


class NeuralProperty(Property):
    """A ``Property`` backed by an Equinox neural network.

    The network receives a normalised 1-D input ``[R / R0]`` and must
    return a 1-D output of shape ``(1,)``.  The caller is responsible for
    any output transform needed to satisfy physical constraints — for
    example, wrapping the final layer output with ``jax.nn.softplus`` or
    ``jnp.exp`` to enforce positivity for surface tension.

    Parameters
    ----------
    net : eqx.Module
        Any callable Equinox module with signature ``(x: Array[1]) -> Array[1]``.
        ``eqx.nn.MLP`` is the natural choice.

    Example
    -------
    >>> import equinox as eqx
    >>> import jax
    >>> import jax.numpy as jnp
    >>> key = jax.random.PRNGKey(0)
    >>> mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=32, depth=3,
    ...                  final_activation=jax.nn.softplus, key=key)
    >>> sigma = NeuralProperty(net=mlp)
    """

    net: eqx.Module

    def __call__(self, state: BubbleState) -> jax.Array:
        x = jnp.array([state.R / state.R0])
        return self.net(x).squeeze()


def as_property(val: float | jax.Array | Property) -> Property:
    """Coerce a plain scalar or JAX array to a ``ConstantProperty``, or pass through.

    Parameters
    ----------
    val : float, jax.Array, or Property
        A plain scalar, a JAX array (possibly a tracer), or an existing
        ``Property`` instance.

    Returns
    -------
    Property
    """
    if isinstance(val, Property):
        return val
    return ConstantProperty(val=val)
