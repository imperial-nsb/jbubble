"""Modular bubble dynamics components.

Compose gas laws, shell coatings, surrounding-medium rheologies, and
equations of motion independently.

This package doesn't re-export its classes, so you import them from their
submodules:

```python
from jbubble.bubble.state import BubbleState
from jbubble.bubble.property import (
    Property,
    ConstantProperty,
    NeuralProperty,
    as_property,
)
from jbubble.bubble.eom import KellerMiksis, RayleighPlesset
from jbubble.bubble.gas import PolytropicGas, VanDerWaalsGas
from jbubble.bubble.shell import LipidShell, MarmottantSurfaceTension
from jbubble.bubble.medium import NewtonianMedium, KelvinVoigtMedium
```
"""

__all__: list[str] = []
