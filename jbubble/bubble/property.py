"""State-dependent property abstractions for bubble models.

A [`Property`][jbubble.bubble.property.Property] is any callable
`eqx.Module` that maps a [`BubbleState`][jbubble.bubble.state.BubbleState]
to a scalar array. This abstraction covers:

- [`ConstantProperty`][jbubble.bubble.property.ConstantProperty]: a fixed
  value (the common case).
- State-dependent laws such as
  [`MarmottantSurfaceTension`][jbubble.bubble.shell.MarmottantSurfaceTension],
  [`SmoothMarmottantSurfaceTension`][jbubble.bubble.shell.SmoothMarmottantSurfaceTension],
  and [`GompertzSurfaceTension`][jbubble.bubble.shell.GompertzSurfaceTension],
  defined in `jbubble.bubble.shell`.
- [`NeuralProperty`][jbubble.bubble.property.NeuralProperty]: a neural
  network that learns an unknown law from data.

Because all `Property` subclasses are Equinox modules, they are full JAX
pytrees: `jit`, `vmap`, and `grad` flow through them without extra
bookkeeping. The [`as_property`][jbubble.bubble.property.as_property]
converter lets model constructors accept plain floats while they still
store a proper `Property` internally.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .state import BubbleState

__all__ = ["Property", "ConstantProperty", "NeuralProperty", "as_property"]


class Property(eqx.Module, abc.ABC):
    """Abstract base for state-dependent (or constant) bubble properties.

    Any callable `eqx.Module` that maps a
    [`BubbleState`][jbubble.bubble.state.BubbleState] to a scalar array
    inherits from this class. The concrete subclass
    [`ConstantProperty`][jbubble.bubble.property.ConstantProperty] handles
    the common case of a fixed value; state-dependent models, such as
    [`MarmottantSurfaceTension`][jbubble.bubble.shell.MarmottantSurfaceTension],
    override `__call__` directly.
    """

    @abc.abstractmethod
    def __call__(self, state: BubbleState) -> jax.Array:
        """Evaluate the property at `state`.

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

    Parameters
    ----------
    val : float or jax.Array
        The constant value. It can be a JAX array, including a traced value
        inside `jax.grad` or `jax.jit`, so gradients flow through it.
    """

    val: ArrayLike

    def __call__(self, state: BubbleState) -> jax.Array:
        return jnp.asarray(self.val) + state.R * 0.0


class NeuralProperty(Property):
    """A [`Property`][jbubble.bubble.property.Property] backed by an Equinox neural network.

    The network receives a normalised 1-D input `[R / R0]` and must
    return a 1-D output of shape `(1,)`. You are responsible for any
    output transform that the physics needs. For example, wrap the final
    layer output with `jax.nn.softplus` or `jnp.exp` to keep a surface
    tension positive.

    Parameters
    ----------
    net : eqx.Module
        Any callable Equinox module with signature
        `(x: Array[1]) -> Array[1]`. `eqx.nn.MLP` is the natural choice.

    Examples
    --------
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
        net_fn = cast(Callable[[jax.Array], jax.Array], self.net)
        return net_fn(x).squeeze()


def as_property(val: ArrayLike | Property) -> Property:
    """Convert a plain scalar or JAX array to a `ConstantProperty`, or pass it through.

    Parameters
    ----------
    val : float, jax.Array, or Property
        A plain scalar, a JAX array (possibly a tracer), or an existing
        [`Property`][jbubble.bubble.property.Property] instance.

    Returns
    -------
    Property
        `val` itself if it's already a `Property`; otherwise
        `ConstantProperty(val=val)`.
    """
    if isinstance(val, Property):
        return val
    return ConstantProperty(val=val)
