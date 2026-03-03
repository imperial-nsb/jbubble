# Pulse shapes

A `Pulse` object wraps a `PulseShape` to define the temporal waveform of the
driving ultrasound pressure. jbubble ships with a large library of shapes
implemented as JAX-compatible modules.

## Constructing a Pulse

```python
from jbubble.pulse import Pulse
from jbubble import shapes

pulse = Pulse(
    freq=2e6,            # carrier frequency in Hz
    pressure=200e3,         # peak negative pressure amplitude in Pa
    cycle_num=3,         # number of oscillation cycles
    shape=shapes.Sine(), # waveform shape
)
```

The `Pulse` multiplies `pressure` by the normalised output of `shape` evaluated at
each time-step. All shapes are normalised so that their peak absolute value is 1.

## Available shapes

### Analytic shapes

| Class | Description |
|---|---|
| `Sine` | Standard sinusoidal waveform — the most common choice |
| `TimeDomainSquare` | Smooth square wave via `tanh(k·sin(ωt))`; sharpness is configurable |
| `TimeDomainSawtooth` | Smooth forward sawtooth via `arctan(tan(ωt))` |
| `TimeDomainTriangle` | Smooth triangle wave via `arcsin(sin(ωt))` |

### Fourier-series shapes

These shapes are constructed from a truncated Fourier series (`NUM_FOURIER_TERMS = 10`).

| Class | Description |
|---|---|
| `Square` | Odd-harmonic Fourier square wave |
| `SquareNegPos` | Square wave: −1 for first half-cycle, +1 for second |
| `Sawtooth` | Forward sawtooth |
| `InvertedSawtooth` | Reverse sawtooth |
| `Triangle` | Symmetric triangle |
| `Quadratic` | Quadratic waveform |
| `NegativeQuadratic` | Inverted quadratic |
| `Asymmetrical` | Mixed cosine+sine asymmetric waveform |
| `SlantedSine` | Sine with harmonic slope (Clausen-function envelope) |
| `Pulse9` | Phase-shifted cosine harmonic series |
| `Pulse10` | Mixed-harmonic series on odd multiples |

### Rectangular duty-cycle shapes

These shapes produce rectangular waves with specified duty cycles.

| Class | Waveform description |
|---|---|
| `Rect25` | +1 for 75% of cycle, −1 for 25% |
| `Rect75` | +1 for 25% of cycle, −1 for 75% |
| `Rect25NegPos` | −1 for first 25%, +1 for last 75% |
| `Rect75NegPos` | −1 for first 75%, +1 for last 25% |
| `Rect95` | −1 for 95% of cycle, +1 for last 5% |
| `Mono95` | −1 for 95% of cycle, 0 for last 5% (monopole-like) |
| `Mono99` | −1 for 99% of cycle, 0 for last 1% (narrow monopole) |

## Creating a custom shape

Subclass `PulseShape` and implement `__call__`:

```python
import jax.numpy as jnp
from jbubble.shapes import PulseShape

class MyShape(PulseShape):
    def __call__(self, t, freq, phase, initial_time):
        t = t - initial_time
        return jnp.sin(2.0 * jnp.pi * freq * t - phase) ** 3

pulse = Pulse(freq=1e6, pressure=100e3, cycle_num=5, shape=MyShape())
```

Custom shapes must be pure JAX functions (no Python control flow on traced values)
to be compatible with `jax.jit` and `jax.vmap`.
