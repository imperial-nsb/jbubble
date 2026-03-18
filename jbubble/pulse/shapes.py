"""Library of differentiable-enough pulse shapes."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

NUM_FOURIER_TERMS = 10


class PulseShape(eqx.Module):
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
        return "base_pulse"


class FourierPulseShape(PulseShape):
    @abc.abstractmethod
    def term(
        self,
        m: jax.Array,
        t: jax.Array,
        freq: ArrayLike,
        phase: ArrayLike,
    ) -> jax.Array:
        pass

    @property
    @abc.abstractmethod
    def norm_factor(self) -> ArrayLike:
        pass

    @property
    def dc_offset(self) -> float:
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
    """General duty-cycle rectangular waveform.

    Parameterized by duty cycle, amplitude levels, and window placement::

        f(t) = high_level  for (phase_offset / 2π) * T ≤ t' < (phase_offset / 2π + duty) * T
        f(t) = low_level   otherwise

    where t' = t mod T.

    DC component: ``high_level * duty + low_level * (1 - duty)``

    Parameters
    ----------
    duty:
        Fraction of the period spent at ``high_level`` (0 < duty < 1).
    high_level:
        Amplitude during the active window (default +1).
    low_level:
        Amplitude outside the active window (default -1).
    phase_offset:
        Cycle offset in radians, equal to ``2π * (start fraction of high window)``.
        ``0`` places the window at the cycle start; ``π`` at the midpoint.

    Examples
    --------
    Common named forms::

        Rectangular(duty=0.5)                                          # ±1 square
        Rectangular(duty=0.5, phase_offset=jnp.pi)                     # NegPos square
        Rectangular(duty=0.25)                                         # +1 for 25%, -1 for 75%
        Rectangular(duty=0.01, high_level=0.0, low_level=-1.0,
                         phase_offset=1.98 * jnp.pi)                       # monopolar 99%
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
