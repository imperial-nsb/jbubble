"""State-dependent material property modules.

The ``Property`` pattern replaces static ``float`` parameters with
explicit Equinox modules whose ``__call__`` maps a ``BubbleState`` to a
scalar value.  This allows users to inject arbitrary non-linear rheology
or even internal neural networks, seamlessly compatible with JAX autodiff.

Surface tension models previously in ``surface.py`` are reimplemented
here as ``Property`` subclasses.
"""

import equinox as eqx
import jax

from .state import BubbleState


class Property(eqx.Module):
    """A property that returns a constant value regardless of state.

    Fields
    ------
    val : float
        The constant value.
    """

    val: float = eqx.field(default_factory=lambda: 0.0, kw_only=True)

    def __call__(self, state: BubbleState) -> jax.Array:
        return self.val + state.R * 0.0


def as_property(val: float | Property) -> "Property":
    """Coerce a plain float to a ``Property``, or pass through a ``Property``.

    Parameters
    ----------
    val : float or Property
        A plain scalar or an existing ``Property`` instance.

    Returns
    -------
    Property
    """
    if isinstance(val, Property):
        return val
    return Property(val=float(val))


