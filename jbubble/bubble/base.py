import abc
import dataclasses

import equinox as eqx
import jax

from .properties import Property, as_property
from .state import BubbleState


class Model(eqx.Module, abc.ABC):
    """Base class for all bubble models."""
    
    @abc.abstractmethod
    def __call__(self, state: BubbleState) -> jax.Array:
        """Evaluate the model at the given state."""
        ...

    def __post_init__(self):
        for f in dataclasses.fields(self):
            if f.type is Property:
                object.__setattr__(
                    self,
                    f.name,
                    as_property(getattr(self, f.name)),
                )
