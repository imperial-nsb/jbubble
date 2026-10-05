r"""Library of carrier waveform shapes for acoustic pulses.

Every shape is differentiable enough for gradient-based work. A shape
evaluates a periodic waveform at the phase $x = 2\pi f (t - t_0) - \phi$.
The Fourier-series shapes sum the first 10 terms of their series.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "PulseShape",
    "FourierPulseShape",
    "Sine",
    "Sawtooth",
    "InvertedSawtooth",
    "Triangle",
    "Quadratic",
    "NegativeQuadratic",
    "Square",
    "TimeDomainSquare",
    "TimeDomainSawtooth",
    "TimeDomainTriangle",
    "Rectangular",
]

NUM_FOURIER_TERMS = 10


class PulseShape(eqx.Module):
    r"""Abstract periodic carrier waveform.

    A pulse calls a shape as `shape(t, freq, phase, initial_time)`. The
    shape returns the waveform value at time `t` [s], evaluated at the
    phase

    $$
    x = 2\pi f (t - t_0) - \phi,
    $$

    where $f$ is `freq` [Hz], $t_0$ is `initial_time` [s], and $\phi$ is
    `phase` [rad].
    """

    @abc.abstractmethod
    def __call__(
        self,
        t: jax.Array,
        freq: ArrayLike,
        phase: ArrayLike,
        initial_time: ArrayLike,
    ) -> jax.Array:
        pass

    @property
    def name(self) -> str:
        """Short identifier for the shape."""
        return "base_pulse"


class FourierPulseShape(PulseShape):
    r"""Abstract carrier waveform built from a truncated Fourier series.

    Subclasses implement `term`, the $m$-th term of the series, and
    `norm_factor`. Term $m$ isn't always harmonic $m$: in
    [`Square`][jbubble.pulse.shapes.Square], term $m$ is harmonic $2m - 1$.
    The shape sums the first 10 terms (`NUM_FOURIER_TERMS`), divides by
    `norm_factor`, and adds `dc_offset`:

    $$
    s(t) = \frac{1}{\mathrm{norm\_factor}}
        \sum_{m=1}^{10} \mathrm{term}(m, t - t_0, f, \phi) + \mathrm{dc\_offset}
    $$
    """

    @abc.abstractmethod
    def term(
        self,
        m: jax.Array,
        t: jax.Array,
        freq: ArrayLike,
        phase: ArrayLike,
    ) -> jax.Array:
        """Return the `m`-th Fourier term at time `t`, relative to `initial_time`."""
        pass

    @property
    @abc.abstractmethod
    def norm_factor(self) -> ArrayLike:
        """Divisor that normalises the summed series."""
        pass

    @property
    def dc_offset(self) -> float:
        """Constant added after normalisation. The base value is `0.0`."""
        return 0.0

    def __call__(
        self,
        t: jax.Array,
        freq: ArrayLike,
        phase: ArrayLike,
        initial_time: ArrayLike,
    ) -> jax.Array:
        t = t - initial_time
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1, dtype=float)

        def term_wrapper(m_val):
            return self.term(m_val, t, freq, phase)

        return (
            jnp.sum(jax.vmap(term_wrapper)(m), axis=0) / self.norm_factor
            + self.dc_offset
        )


class Sine(PulseShape):
    r"""Pure sine carrier.

    $$
    s = \sin x,
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape].
    """

    def __call__(
        self,
        t: jax.Array,
        freq: ArrayLike,
        phase: ArrayLike,
        initial_time: ArrayLike,
    ) -> jax.Array:
        t = t - initial_time
        return jnp.sin(2.0 * jnp.pi * freq * t - phase)

    @property
    def name(self) -> str:
        return "sine"


class Sawtooth(FourierPulseShape):
    r"""Rising sawtooth from a 10-term Fourier series.

    $$
    s = \frac{2}{\pi} \sum_{m=1}^{10} \frac{(-1)^{m+1}}{m} \sin(m x),
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape].
    """

    def term(
        self, m: jax.Array, t: jax.Array, freq: ArrayLike, phase: ArrayLike
    ) -> jax.Array:
        return -((-1) ** m / m) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)

    @property
    def norm_factor(self) -> ArrayLike:
        return jnp.pi / 2.0

    @property
    def name(self) -> str:
        return "sawtooth"


class InvertedSawtooth(FourierPulseShape):
    r"""Falling sawtooth, the negative of [`Sawtooth`][jbubble.pulse.shapes.Sawtooth].

    $$
    s = \frac{2}{\pi} \sum_{m=1}^{10} \frac{(-1)^{m}}{m} \sin(m x),
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape].
    """

    def term(
        self, m: jax.Array, t: jax.Array, freq: ArrayLike, phase: ArrayLike
    ) -> jax.Array:
        return ((-1) ** m / m) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)

    @property
    def norm_factor(self) -> ArrayLike:
        return jnp.pi / 2.0

    @property
    def name(self) -> str:
        return "inverted_sawtooth"


class Triangle(FourierPulseShape):
    r"""Triangle wave from a 10-term Fourier series (odd harmonics only).

    $$
    s = -\frac{4}{\pi^2} \sum_{m=1}^{10} \frac{1 - (-1)^m}{m^2}
        \cos\left(m\left(x + \frac{\pi}{2}\right)\right),
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape]. The wave starts at 0
    and peaks at $x = \pi/2$.
    """

    def term(
        self, m: jax.Array, t: jax.Array, freq: ArrayLike, phase: ArrayLike
    ) -> jax.Array:
        return -((1 - (-1) ** m) / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * (t + (1.0 / (4.0 * freq))) - m * phase
        )

    @property
    def norm_factor(self) -> ArrayLike:
        return (jnp.pi**2) / 4.0

    @property
    def name(self) -> str:
        return "triangle"


class Quadratic(FourierPulseShape):
    r"""Piecewise-parabolic wave from a 10-term Fourier series.

    $$
    s = \frac{6}{\pi^2} \sum_{m=1}^{10} \frac{(-1)^m}{m^2}
        \cos\left(m\left(x - \frac{\pi}{\sqrt{3}}\right)\right),
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape].
    """

    def term(
        self, m: jax.Array, t: jax.Array, freq: ArrayLike, phase: ArrayLike
    ) -> jax.Array:
        p = jnp.pi / jnp.sqrt(3.0)
        return ((-1) ** m / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * t - m * phase - m * p
        )

    @property
    def norm_factor(self) -> ArrayLike:
        return (jnp.pi**2) / 6.0

    @property
    def name(self) -> str:
        return "quadratic"


class NegativeQuadratic(Quadratic):
    """Negative of [`Quadratic`][jbubble.pulse.shapes.Quadratic]."""

    def __call__(
        self,
        t: jax.Array,
        freq: ArrayLike,
        phase: ArrayLike,
        initial_time: ArrayLike,
    ) -> jax.Array:
        return -super().__call__(t, freq, phase, initial_time)

    @property
    def name(self) -> str:
        return "negative_quadratic"


class Square(FourierPulseShape):
    r"""Square wave from a 10-term Fourier series (odd harmonics 1 to 19).

    $$
    s = \frac{4}{\pi} \sum_{m=1}^{10} \frac{\sin\left((2m - 1) x\right)}{2m - 1},
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape]. The truncated series
    shows Gibbs ringing at the edges.
    """

    def term(
        self, m: jax.Array, t: jax.Array, freq: ArrayLike, phase: ArrayLike
    ) -> jax.Array:
        return (1.0 / (2 * m - 1)) * jnp.sin(
            2.0 * jnp.pi * (2 * m - 1) * freq * t - (2 * m - 1) * phase
        )

    @property
    def norm_factor(self) -> ArrayLike:
        return jnp.pi / 4.0

    @property
    def name(self) -> str:
        return "square"


class TimeDomainSquare(PulseShape):
    r"""Smooth square wave built in the time domain.

    $$
    s = \tanh(k \sin x),
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape], and $k$ is `sharpness`.

    Parameters
    ----------
    sharpness : float
        Steepness $k$ of the transitions. Default: `50.0`.
    """

    sharpness: float = 50.0

    def __call__(
        self,
        t: jax.Array,
        freq: ArrayLike,
        phase: ArrayLike,
        initial_time: ArrayLike,
    ) -> jax.Array:
        t = t - initial_time
        return jnp.tanh(self.sharpness * jnp.sin(2.0 * jnp.pi * freq * t - phase))

    @property
    def name(self) -> str:
        return "time_domain_square"


class TimeDomainSawtooth(PulseShape):
    r"""Rising sawtooth built in the time domain.

    $$
    s = \frac{2}{\pi} \arctan\left(\tan\frac{x}{2}\right),
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape].
    """

    def __call__(
        self,
        t: jax.Array,
        freq: ArrayLike,
        phase: ArrayLike,
        initial_time: ArrayLike,
    ) -> jax.Array:
        t = t - initial_time
        return (2.0 / jnp.pi) * jnp.arctan(jnp.tan(jnp.pi * freq * t - phase / 2.0))

    @property
    def name(self) -> str:
        return "time_domain_sawtooth"


class TimeDomainTriangle(PulseShape):
    r"""Triangle wave built in the time domain.

    $$
    s = \frac{2}{\pi} \arcsin(\sin x),
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape].
    """

    def __call__(
        self,
        t: jax.Array,
        freq: ArrayLike,
        phase: ArrayLike,
        initial_time: ArrayLike,
    ) -> jax.Array:
        t = t - initial_time
        return (2.0 / jnp.pi) * jnp.arcsin(jnp.sin(2.0 * jnp.pi * freq * t - phase))

    @property
    def name(self) -> str:
        return "time_domain_triangle"


class Rectangular(FourierPulseShape):
    r"""General duty-cycle rectangular waveform.

    Parameterised by the duty cycle $D$, the amplitude levels $A$ and $B$,
    and the window placement $\phi_\text{off}$. The target waveform is

    $$
    s =
    \begin{cases}
    A & \left(t - t_0 - \dfrac{\phi_\text{off}}{2\pi} T\right) \bmod T < D T, \\
    B & \text{otherwise},
    \end{cases}
    $$

    for zero carrier phase, where $T = 1/f$ is the period. The high window
    starts a fraction $\phi_\text{off}/2\pi$ into each period and lasts
    $DT$; when $\phi_\text{off}/2\pi + D > 1$, it wraps into the start of
    the next period. The code evaluates the first 10 harmonics of its
    Fourier series:

    $$
    \begin{aligned}
    s &= A D + B (1 - D) + \sum_{m=1}^{10} \left[a_m \cos(m y) + b_m \sin(m y)\right],
    \qquad y = x - \phi_\text{off}, \\
    a_m &= \frac{A - B}{\pi m} \sin(2\pi m D),
    \qquad
    b_m = \frac{A - B}{\pi m} \left[1 - \cos(2\pi m D)\right],
    \end{aligned}
    $$

    with $x = 2\pi f (t - t_0) - \phi$ as in
    [`PulseShape`][jbubble.pulse.shapes.PulseShape]. The DC component is
    `high_level * duty + low_level * (1 - duty)`.

    Parameters
    ----------
    duty : float
        Fraction $D$ of the period spent at `high_level`, with
        0 < `duty` < 1.
    high_level : float
        Amplitude $A$ during the active window. Default: `1.0`.
    low_level : float
        Amplitude $B$ outside the active window. Default: `-1.0`.
    phase_offset : float
        Cycle offset $\phi_\text{off}$ [rad], equal to $2\pi$ times the
        start fraction of the high window. `0` places the window at the
        cycle start and `π` at the midpoint. Default: `0.0`.

    Examples
    --------
    The following examples show common named forms.

    ```python
    Rectangular(duty=0.5)  # ±1 square
    Rectangular(duty=0.5, phase_offset=jnp.pi)  # negative-then-positive square
    Rectangular(duty=0.25)  # +1 for 25% of the cycle, -1 for 75%
    Rectangular(  # monopolar: -1 for 99% of the cycle
        duty=0.01, high_level=0.0, low_level=-1.0, phase_offset=1.98 * jnp.pi
    )
    ```
    """

    duty: float
    high_level: float = 1.0
    low_level: float = -1.0
    phase_offset: float = 0.0

    def term(
        self, m: jax.Array, t: jax.Array, freq: ArrayLike, phase: ArrayLike
    ) -> jax.Array:
        D = self.duty
        A = self.high_level
        B = self.low_level
        omega = 2.0 * jnp.pi * freq

        factor = (A - B) / (jnp.pi * m)
        a_m = factor * jnp.sin(2.0 * jnp.pi * m * D)
        b_m = factor * (1.0 - jnp.cos(2.0 * jnp.pi * m * D))

        angle = m * (omega * t - (phase + self.phase_offset))
        return a_m * jnp.cos(angle) + b_m * jnp.sin(angle)

    @property
    def norm_factor(self) -> float:
        return 1.0

    @property
    def dc_offset(self) -> float:
        return self.high_level * self.duty + self.low_level * (1.0 - self.duty)

    @property
    def name(self) -> str:
        return f"rect_d{self.duty:.2f}"
