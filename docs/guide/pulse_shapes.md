# Pulse shapes

An acoustic pulse in jbubble is a callable `p_ac(t)` returning the driving pressure in Pascals at time `t`. Every pulse is an Equinox module (a JAX PyTree), so pulses are JIT-compilable and differentiable.

---

## ToneBurst

The standard ultrasound pulse: a carrier waveform gated by a finite-duration envelope.

```python
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine

pulse = ToneBurst(
    freq=1e6,        # centre frequency [Hz]
    pressure=100e3,  # peak pressure amplitude [Pa]
    shape=Sine(),    # carrier waveform
    cycle_num=5,     # number of complete cycles
    # Optional:
    # phase=0.0      # initial carrier phase [rad]
    # envelope=SoftRectangularEnvelope()
)
```

### Carrier shapes

The `shape` argument selects the carrier waveform within each cycle. All shapes are normalised to peak amplitude 1.

| Shape class | Description |
|---|---|
| `Sine` | Standard sine wave (default) |
| `Square` | Fourier-series square wave (10 terms) |
| `Sawtooth` | Rising sawtooth |
| `InvertedSawtooth` | Falling sawtooth |
| `Triangle` | Triangular wave |
| `Quadratic` | Parabolic carrier |
| `NegativeQuadratic` | Inverted parabolic carrier |
| `Rectangular(duty)` | General duty-cycle rectangular waveform |
| `TimeDomainSquare(sharpness)` | `tanh`-smoothed square, avoids Gibbs ringing |
| `TimeDomainSawtooth` | `arctan`-based smooth sawtooth |
| `TimeDomainTriangle` | `arcsin`-based smooth triangle |

```python
from jbubble.pulse.shapes import Square, Rectangular
import jax.numpy as jnp

pulse = ToneBurst(freq=1e6, pressure=200e3, shape=Square())

# Monopolar rectangular pulse (99% duty)
pulse = ToneBurst(
    freq=1e6, pressure=200e3,
    shape=Rectangular(duty=0.01, high_level=0.0, low_level=-1.0,
                      phase_offset=1.98 * jnp.pi),
)
```

### Envelopes

The envelope gates the carrier over the active window. The default for `ToneBurst` is `SoftRectangularEnvelope`.

| Envelope class | Description |
|---|---|
| `RectangularEnvelope` | Hard on/off step. **Avoid for fitting** — non-differentiable edges. |
| `SoftRectangularEnvelope` | Sigmoid on and off transitions. $C^\infty$, near-rectangular. **Preferred.** |
| `HannEnvelope` | Cosine-squared window. Smooth, tapered edges. |
| `TukeyEnvelope(alpha)` | Hann-tapered at both ends, flat in the middle. `alpha` controls taper fraction. |

```python
from jbubble.pulse import ToneBurst, HannEnvelope, TukeyEnvelope
from jbubble.pulse.shapes import Sine

# Hann-windowed tone burst
pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5,
                  envelope=HannEnvelope())

# 10% taper on each side, flat in the middle
pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=20,
                  envelope=TukeyEnvelope(alpha=0.1))
```

---

## ChirpPulse

A frequency-swept pulse. Useful for broadband excitation and some therapeutic protocols.

```python
from jbubble.pulse import ChirpPulse

pulse = ChirpPulse(
    freq_start=0.5e6,   # start frequency [Hz]
    freq_end=2.0e6,     # end frequency [Hz]
    pressure=100e3,     # peak amplitude [Pa]
    duration=20e-6,     # pulse duration [s]
    # mode="linear" or "exponential"
)
```

---

## SampledPulse

Wraps a discrete pressure waveform, interpolating at query times. Useful when the driving waveform is measured experimentally.

```python
import jax.numpy as jnp
from jbubble.pulse import SampledPulse

ts = jnp.linspace(0, 10e-6, 1000)   # time axis [s]
ps = measured_waveform               # pressure values [Pa], shape (1000,)

pulse = SampledPulse(ts=ts, ps=ps)
```

The interpolation is performed with `jnp.interp` (linear, clamped to boundary values outside the range), so it is differentiable through the sampled pressures.

---

## NeuralPulse

A pulse parameterised by a small neural network. Useful for learned or optimised driving waveforms.

```python
import equinox as eqx
from jbubble.pulse import NeuralPulse

net = eqx.nn.MLP(1, 1, width_size=32, depth=3, key=jax.random.key(0))
pulse = NeuralPulse(net=net)
```

The network receives the scalar time `t` and returns the driving pressure. Because it is an Equinox module, the network weights are differentiable and can be optimised via `fit_parameters`.

---

## Composing pulses

### Superposition

```python
from jbubble.pulse import ToneBurst, Summed
from jbubble.pulse.shapes import Sine

# Dual-frequency driving
p1 = ToneBurst(freq=1e6, pressure=80e3, shape=Sine(), cycle_num=10)
p2 = ToneBurst(freq=2e6, pressure=40e3, shape=Sine(), cycle_num=20)
pulse = Summed(p1, p2)
```

### Amplitude scaling

```python
from jbubble.pulse import Scaled

pulse = Scaled(base=p1, scale=2.0)  # doubles the amplitude
```

### Time offset

```python
from jbubble.pulse import Offset

delayed_pulse = Offset(base=p1, offset=5e-6)  # 5 µs delay
```

### Applying an envelope to any pulse

Every `Pulse` has a `.windowed(envelope)` method:

```python
from jbubble.pulse import SoftRectangularEnvelope

windowed = sampled_pulse.windowed(SoftRectangularEnvelope())
```

---

## Evaluating a pulse

All pulses share the same interface:

```python
import jax.numpy as jnp

t = jnp.linspace(0, 10e-6, 1000)
p = jax.vmap(pulse)(t)  # shape (1000,) [Pa]
```

Or simply call at a scalar time:

```python
p_at_3us = pulse(3e-6)
```
