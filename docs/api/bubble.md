# bubble

The `jbubble.bubble` subpackage contains all bubble physics components: state representation, the Property abstraction, and gas/shell/medium/EoM models.

---

## State

::: jbubble.bubble.state.BubbleState

::: jbubble.bubble.state.ConfinedBubbleState

---

## Properties

The `Property` abstraction is the key extensibility mechanism in jbubble. Any physical parameter that *could* depend on the bubble state is represented as a `Property` — a callable `state → scalar`. Plain floats are automatically promoted via `as_property`.

::: jbubble.bubble.property.Property

::: jbubble.bubble.property.ConstantProperty

::: jbubble.bubble.property.NeuralProperty

::: jbubble.bubble.property.as_property

---

## Gas models

::: jbubble.bubble.gas.GasModel

::: jbubble.bubble.gas.PolytropicGas

::: jbubble.bubble.gas.VanDerWaalsGas

---

## Shell models

::: jbubble.bubble.shell.ShellModel

::: jbubble.bubble.shell.NoShell

::: jbubble.bubble.shell.LipidShell

::: jbubble.bubble.shell.ThickShell

### Surface tension Properties

These are `Property` subclasses that encode a state-dependent surface tension law. They are passed as the `sigma` argument to shell models.

::: jbubble.bubble.shell.MarmottantSurfaceTension

::: jbubble.bubble.shell.GompertzSurfaceTension

---

## Medium models

::: jbubble.bubble.medium.MediumModel

::: jbubble.bubble.medium.NewtonianMedium

::: jbubble.bubble.medium.KelvinVoigtMedium

::: jbubble.bubble.medium.NeoHookeanMedium

::: jbubble.bubble.medium.PowerLawMedium

---

## Equations of motion

::: jbubble.bubble.eom.EquationOfMotion

::: jbubble.bubble.eom.RayleighPlesset

::: jbubble.bubble.eom.ModifiedRayleighPlesset

::: jbubble.bubble.eom.KellerMiksis

::: jbubble.bubble.eom.Gilmore

::: jbubble.bubble.eom.LeightonTube

::: jbubble.bubble.eom.SphericalConfinement
