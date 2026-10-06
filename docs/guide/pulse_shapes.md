# Pulse shapes

A pulse is the acoustic pressure $p_\text{ac}(t)$ that drives the bubble.
Every pulse is a callable [Equinox](https://docs.kidger.site/equinox/)
module: `pulse(t)` returns the pressure in pascals at time `t` in seconds.
Because pulses are JAX PyTrees, you can compile, batch, and differentiate
through their parameters, such as the amplitude, the frequency, or the
weights of a neural network.

Every code block on this page runs as written, in order.

## Tone bursts

A [`ToneBurst`][jbubble.pulse.tone_burst.ToneBurst] is a periodic carrier
shape, multiplied by a peak pressure and gated by an envelope:

$$
p(t) = P\, s(t - t_0; f, \phi)\, w(t - t_0, T), \qquad T = \frac{N}{f},
$$

where $P$ is `pressure`, $s$ is `shape`, $f$ is `freq`, $\phi$ is `phase`,
$t_0$ is `initial_time`, $N$ is `cycle_num`, and $w$ is `envelope`.

```python
import jax
import jax.numpy as jnp

from jbubble.pulse import HannEnvelope, ToneBurst
from jbubble.pulse.shapes import Sine

pulse = ToneBurst(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)
hann = ToneBurst(
    freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5, envelope=HannEnvelope()
)

ts = jnp.linspace(0.0, 6e-6, 601)  # [s]
p = jax.vmap(pulse)(ts)  # [Pa]
print(
    f"peak {p.max() / 1e3:.1f} kPa, Hann peak {jax.vmap(hann)(ts).max() / 1e3:.1f} kPa"
)
```

### Carrier shapes

Shapes from `jbubble.pulse.shapes` have a nominal amplitude of 1.

| Shape | Waveform |
|---|---|
| [`Sine`][jbubble.pulse.shapes.Sine] | Pure sine |
| [`Square`][jbubble.pulse.shapes.Square] | Square wave from a 10-term Fourier series, with Gibbs ringing at the edges |
| [`Sawtooth`][jbubble.pulse.shapes.Sawtooth], [`InvertedSawtooth`][jbubble.pulse.shapes.InvertedSawtooth] | Rising or falling sawtooth from a 10-term Fourier series |
| [`Triangle`][jbubble.pulse.shapes.Triangle] | Triangle wave from a 10-term Fourier series |
| [`Quadratic`][jbubble.pulse.shapes.Quadratic], [`NegativeQuadratic`][jbubble.pulse.shapes.NegativeQuadratic] | Piecewise-parabolic wave and its negative |
| [`Rectangular`][jbubble.pulse.shapes.Rectangular] | Rectangular wave with a duty cycle and two levels, such as a monopolar pulse train |
| [`TimeDomainSquare`][jbubble.pulse.shapes.TimeDomainSquare], [`TimeDomainSawtooth`][jbubble.pulse.shapes.TimeDomainSawtooth], [`TimeDomainTriangle`][jbubble.pulse.shapes.TimeDomainTriangle] | Smooth shapes built in the time domain, without Fourier ringing |

```{.python continuation}
from jbubble.pulse.shapes import Rectangular, TimeDomainSquare

square = ToneBurst(freq=1e6, pressure=100e3, shape=TimeDomainSquare(), cycle_num=5)
# Negative half-cycle pulses: -1 for 10 % of each period, 0 otherwise.
monopolar = ToneBurst(
    freq=1e6,
    pressure=100e3,
    shape=Rectangular(duty=0.1, high_level=-1.0, low_level=0.0),
    cycle_num=5,
)
```

### Envelopes

The envelope gates the carrier to the active window. A single pulse, such as
a tone burst, defaults to
[`SoftRectangularEnvelope`][jbubble.pulse.envelope.SoftRectangularEnvelope],
and a sum of pulses to
[`NoEnvelope`][jbubble.pulse.envelope.NoEnvelope].

| Envelope | Window |
|---|---|
| [`SoftRectangularEnvelope`][jbubble.pulse.envelope.SoftRectangularEnvelope] | Near-rectangular, with smooth sigmoid edges: the default, and the best choice for gradients |
| [`RectangularEnvelope`][jbubble.pulse.envelope.RectangularEnvelope] | Hard on and off steps; avoid it when you differentiate, because the drive's slope jumps at the edges |
| [`HannEnvelope`][jbubble.pulse.envelope.HannEnvelope] | Raised cosine: smooth, with a narrow spectrum |
| [`TukeyEnvelope`][jbubble.pulse.envelope.TukeyEnvelope] | Flat in the middle with cosine tapers; `alpha` is the fraction of the window in the tapers |
| [`NoEnvelope`][jbubble.pulse.envelope.NoEnvelope] | 1 at all times: the default of a sum of pulses |

To change the envelope of an existing pulse, call
[`windowed`][jbubble.pulse.base.Pulse.windowed]:

```{.python continuation}
from jbubble.pulse import TukeyEnvelope

tapered = pulse.windowed(TukeyEnvelope(alpha=0.2))
```

## Timing: start, stop, and end

Each pulse has an active window, from
[`t_start`][jbubble.pulse.base.Pulse.t_start] to
[`t_stop`][jbubble.pulse.base.Pulse.t_stop]. To delay a pulse, set its
keyword-only `initial_time`. [`t_end`][jbubble.pulse.base.Pulse.t_end],
the stop time of a simulation that doesn't set `t_max`, is the start time
plus twice the duration, so the bubble has time to ring down:

```{.python continuation}
delayed = ToneBurst(
    freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5, initial_time=2e-6
)
print(f"start {delayed.t_start * 1e6:.1f} µs, stop {delayed.t_stop * 1e6:.1f} µs, "
      f"end {delayed.t_end * 1e6:.1f} µs")
```

The solver steps to every edge in
[`window_edges`][jbubble.pulse.base.Pulse.window_edges], so it can't step
over a pulse that starts late.

## Chirps

A [`ChirpPulse`][jbubble.pulse.chirp.ChirpPulse] sweeps its frequency from
`freq_start` to `freq_end` over `sweep_duration`. The sweep law is linear by
default; [`ExponentialSweep`][jbubble.pulse.chirp.ExponentialSweep] sweeps
geometrically, with equal time per octave:

```{.python continuation}
from jbubble.pulse import ChirpPulse
from jbubble.pulse.chirp import ExponentialSweep

chirp = ChirpPulse(
    freq_start=0.5e6,  # [Hz]
    freq_end=3e6,  # [Hz]
    pressure=50e3,  # [Pa]
    sweep_duration=10e-6,  # [s]
    sweep=ExponentialSweep(),
    envelope=TukeyEnvelope(alpha=0.2),
)
```

## Measured waveforms

A [`SampledPulse`][jbubble.pulse.sampled.SampledPulse] interpolates a
sampled pressure trace, such as a hydrophone recording, linearly between the
samples. The sample times are absolute: the active window runs from the
first sample to the last. The following code builds a stand-in for a
measured trace, sampled at 100 MHz:

```{.python continuation}
from jbubble.pulse import RectangularEnvelope, SampledPulse

dt = 10e-9  # 100 MHz sampling [s]
t_rec = jnp.arange(800) * dt
measured = 80e3 * jnp.sin(2 * jnp.pi * 1e6 * t_rec) * jnp.exp(
    -(((t_rec - 4e-6) / 1.5e-6) ** 2)
)  # stand-in for a hydrophone recording [Pa]

recorded = SampledPulse.from_uniform(measured, dt=dt)
print(f"window: {recorded.t_start * 1e6:.2f} to {recorded.t_stop * 1e6:.2f} µs")
```

The default soft envelope halves the first and last samples. If the trace
already starts and ends at zero, as here, that changes nothing. Otherwise,
pass `envelope=RectangularEnvelope()` to keep them, or
`envelope=HannEnvelope()` to taper a trace that's cut off mid-signal.

## Neural pulses

A [`NeuralPulse`][jbubble.pulse.neural.NeuralPulse] lets a neural network
define the waveform: the network maps the normalised time
$(t - t_0)/T$, from 0 to 1 across the window, to the pressure in units of
`pressure_scale`. Its weights are parameters that you can optimise, for
example to design a drive that maximises a bubble response:

```{.python continuation}
import equinox as eqx

from jbubble.pulse import NeuralPulse

net = eqx.nn.MLP(in_size=1, out_size=1, width_size=32, depth=2, key=jax.random.key(0))
learned = NeuralPulse(net=net, pulse_duration=5e-6, pressure_scale=100e3)
```

`pulse_duration` and `pressure_scale` are fixed configuration, not
trainable parameters. To scale a neural pulse by a traced value, multiply
it, as in the next section.

## Combine pulses

Pulses support `+` and `*`:

| Expression | Result | Meaning |
|---|---|---|
| `pulse_a + pulse_b` | [`Summed`][jbubble.pulse.base.Summed] | Superposition, such as dual-frequency driving |
| `k * pulse` | [`Scaled`][jbubble.pulse.base.Scaled] | Amplitude scaling by a number or a JAX scalar |
| `pulse + c` | [`Offset`][jbubble.pulse.base.Offset] | A constant pressure `c` added at all times, not a time delay |

```{.python continuation}
low = ToneBurst(freq=1e6, pressure=80e3, shape=Sine(), cycle_num=10)
high = ToneBurst(freq=2e6, pressure=40e3, shape=Sine(), cycle_num=20)
dual = low + high  # Summed((low, high))
half = 0.5 * dual  # Scaled
smooth_dual = dual.windowed(HannEnvelope())  # windows the sum
print(type(dual).__name__, type(half).__name__, f"{float(dual.duration) * 1e6:.0f} µs")
```

Each part of a sum keeps its own envelope and timing, so a delayed part
starts on time. A sum has no window of its own until you call `windowed`.

## Simulate with a pulse and differentiate it

Any pulse drives any equation of motion. Because the pulse is part of the
model, `jax.grad` differentiates a simulated quantity with respect to its
parameters. The following code computes how the peak radius of a 2 µm
lipid-coated bubble responds to the drive amplitude:

```{.python continuation}
from jbubble import run_simulation
from jbubble.utils.presets import lipid_bubble

eom, _ = lipid_bubble()


def peak_ratio(k):
    result = run_simulation(eom, k * pulse)
    return result.radius.max() / eom.R0


value, slope = jax.value_and_grad(peak_ratio)(1.0)
print(f"peak R/R0 = {value:.3f}, d(peak R/R0)/dk = {slope:.3f}")
```

For the pulse shapes, the algebra, and simulations under each kind of
pulse, with plots, see the example
[Driving pulses](../examples/02_driving_pulses.md).
