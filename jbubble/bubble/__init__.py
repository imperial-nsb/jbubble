"""
Modular bubble dynamics components.

Compose gas laws, shell coatings, surrounding-medium rheologies, and
equations of motion independently.

Import from specific submodules:
  from jbubble.bubble.state    import BubbleState, ConfinedBubbleState
  from jbubble.bubble.property import Property, ConstantProperty, NeuralProperty, as_property
  from jbubble.bubble.eom      import KellerMiksis, RayleighPlesset
  from jbubble.bubble.gas      import PolytropicGas, VanDerWaalsGas
  from jbubble.bubble.shell    import LipidShell, MarmottantSurfaceTension
  from jbubble.bubble.medium   import NewtonianMedium, KelvinVoigtMedium
"""
