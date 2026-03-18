# shapes

Carrier waveform shapes for use with `ToneBurst`. All shapes are `PulseShape` subclasses — Equinox modules with a normalised periodic output (peak amplitude 1).

Shapes are passed as the `shape` argument to `ToneBurst`:

```python
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine, Square, Rectangular

pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine())
```

---

## Base classes

::: jbubble.pulse.shapes.PulseShape

::: jbubble.pulse.shapes.FourierPulseShape

---

## Concrete shapes

::: jbubble.pulse.shapes.Sine

::: jbubble.pulse.shapes.Square

::: jbubble.pulse.shapes.Sawtooth

::: jbubble.pulse.shapes.InvertedSawtooth

::: jbubble.pulse.shapes.Triangle

::: jbubble.pulse.shapes.Quadratic

::: jbubble.pulse.shapes.NegativeQuadratic

::: jbubble.pulse.shapes.Rectangular

::: jbubble.pulse.shapes.TimeDomainSquare

::: jbubble.pulse.shapes.TimeDomainSawtooth

::: jbubble.pulse.shapes.TimeDomainTriangle

---

## Choosing a shape

| Shape | Spectral content | Notes |
|---|---|---|
| `Sine` | Single harmonic | Default; smoothest, no harmonic artefacts |
| `Square` / `TimeDomainSquare` | Odd harmonics | Gibbs ringing in Fourier version; `TimeDomainSquare` avoids it |
| `Sawtooth` / `TimeDomainSawtooth` | All harmonics | Rising ramp; useful for asymmetric forcing |
| `Triangle` / `TimeDomainTriangle` | Odd harmonics, faster roll-off | Smoother than square |
| `Rectangular(duty)` | General | Full control over duty cycle, levels, and phase offset |

For gradient-based optimisation of the carrier shape, prefer `TimeDomain*` variants or `Sine` — Fourier-series shapes truncate at 10 terms and are `C^0` at sharp transitions.
