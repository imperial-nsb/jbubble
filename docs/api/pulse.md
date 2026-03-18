# pulse

The `jbubble.pulse` module provides composable, differentiable acoustic driving waveforms.

All pulses are Equinox modules and implement the interface:

```python
pressure = pulse(t)   # scalar time → scalar pressure [Pa]
```

---

## Base classes

::: jbubble.pulse.base.Pulse

::: jbubble.pulse.base.Scaled

::: jbubble.pulse.base.Summed

::: jbubble.pulse.base.Offset

---

## Pulse types

::: jbubble.pulse.tone_burst.ToneBurst

::: jbubble.pulse.chirp.ChirpPulse

::: jbubble.pulse.sampled.SampledPulse

::: jbubble.pulse.neural.NeuralPulse

---

## Envelopes

::: jbubble.pulse.envelope.Envelope

::: jbubble.pulse.envelope.RectangularEnvelope

::: jbubble.pulse.envelope.SoftRectangularEnvelope

::: jbubble.pulse.envelope.HannEnvelope

::: jbubble.pulse.envelope.TukeyEnvelope
