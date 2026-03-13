import abc
import dataclasses

import equinox as eqx
import jax

from .properties import as_property
from .state import BubbleState


class Model(eqx.Module, abc.ABC):
    """Base class for all bubble models."""
    
    @abc.abstractmethod
    def __call__(self, state: BubbleState) -> jax.Array:
        """Evaluate the model at the given state."""
        ...

    def __post_init__(self):
        for f in dataclasses.fields(self):
            scale = f.metadata.get("property_scale")
            if scale is not None:
                object.__setattr__(
                    self,
                    f.name,
                    as_property(getattr(self, f.name), scale),
                )
