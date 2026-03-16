"""Library of differentiable-enough pulse shapes."""

from __future__ import annotations


import abc

import equinox as eqx
import jax
import jax.numpy as jnp

NUM_FOURIER_TERMS = 10

# Analytical normalization factors are based on infinite series limits or coefficient sums
# For divergent series, we normalize by the sum of absolute coefficients


class PulseShape(eqx.Module):
    @abc.abstractmethod
    def __call__(
        self, t: jax.Array, freq: float, phase: float, initial_time: float
    ) -> jax.Array:
        pass

    @property
    def name(self) -> str:
        return "base_pulse"


class FourierPulseShape(PulseShape):
    @abc.abstractmethod
    def term(self, m: jax.Array, t: jax.Array, freq: float, phase: float) -> jax.Array:
        pass

    @property
    @abc.abstractmethod
    def norm_factor(self) -> float | jax.Array:
        pass

    def __call__(self, t, freq, phase, initial_time):

        t = t - initial_time
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

        def term_wrapper(m_val):
            return self.term(m_val, t, freq, phase)

        return jnp.sum(jax.vmap(term_wrapper)(m), axis=0) / self.norm_factor


class Sine(PulseShape):
    def __call__(self, t, freq, phase, initial_time):
        t = t - initial_time
        return jnp.sin(2.0 * jnp.pi * freq * t - phase)

    @property
    def name(self) -> str:
        return "sine"


class Sawtooth(FourierPulseShape):
    def term(self, m, t, freq, phase):
        return -((-1) ** m / m) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)

    @property
    def norm_factor(self) -> float | jax.Array:
        return jnp.pi / 2.0

    @property
    def name(self) -> str:
        return "sawtooth"


class InvertedSawtooth(FourierPulseShape):
    def term(self, m, t, freq, phase):
        # Flip the sign of the Fourier expansion
        return ((-1) ** m / m) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)

    @property
    def norm_factor(self) -> float | jax.Array:
        return jnp.pi / 2.0

    @property
    def name(self) -> str:
        return "inverted_sawtooth"


class Triangle(FourierPulseShape):
    def term(self, m, t, freq, phase):
        return -((1 - (-1) ** m) / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * (t + (1.0 / (4.0 * freq))) - m * phase
        )

    @property
    def norm_factor(self) -> float | jax.Array:
        return (jnp.pi**2) / 4.0

    @property
    def name(self) -> str:
        return "triangle"


class Quadratic(FourierPulseShape):
    def term(self, m, t, freq, phase):
        p = jnp.pi / jnp.sqrt(3.0)
        return ((-1) ** m / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * t - m * phase - m * p
        )

    @property
    def norm_factor(self) -> float | jax.Array:
        return (jnp.pi**2) / 6.0

    @property
    def name(self) -> str:
        return "quadratic"


class NegativeQuadratic(Quadratic):
    def __call__(self, t, freq, phase, initial_time):
        return -super().__call__(t, freq, phase, initial_time)

    @property
    def name(self) -> str:
        return "negative_quadratic"


class Asymmetrical(FourierPulseShape):
    def term(self, m, t, freq, phase):
        return (1.0 / (m**2)) * jnp.cos(2.0 * jnp.pi * m * freq * t - m * phase) - (
            1.0 / m
        ) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)

    @property
    def norm_factor(self) -> float | jax.Array:
        return -((jnp.pi**2) / 6.0 + jnp.pi / 2.0)

    @property
    def name(self) -> str:
        return "asymmetrical"


class SlantedSine(FourierPulseShape):
    def term(self, m, t, freq, phase):
        return ((-1) ** m / (m**2)) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)

    @property
    def norm_factor(self) -> float | jax.Array:
        return 1.0149  # Approx max of Clausen function Cl2

    @property
    def name(self) -> str:
        return "slanted_sine"


class Square(FourierPulseShape):
    def term(self, m, t, freq, phase):
        return (1.0 / (2 * m - 1)) * jnp.sin(
            2.0 * jnp.pi * (2 * m - 1) * freq * t - (2 * m - 1) * phase
        )

    @property
    def norm_factor(self) -> float | jax.Array:
        return jnp.pi / 4.0

    @property
    def name(self) -> str:
        return "square"


class TimeDomainSquare(PulseShape):
    sharpness: float = 50.0

    def __call__(self, t, freq, phase, initial_time):
        t = t - initial_time
        return jnp.tanh(self.sharpness * jnp.sin(2.0 * jnp.pi * freq * t - phase))

    @property
    def name(self) -> str:
        return "time_domain_square"


class TimeDomainSawtooth(PulseShape):
    def __call__(self, t, freq, phase, initial_time):
        t = t - initial_time
        return (2.0 / jnp.pi) * jnp.arctan(jnp.tan(jnp.pi * freq * t - phase / 2.0))

    @property
    def name(self) -> str:
        return "time_domain_sawtooth"


class TimeDomainTriangle(PulseShape):
    def __call__(self, t, freq, phase, initial_time):
        t = t - initial_time
        return (2.0 / jnp.pi) * jnp.arcsin(jnp.sin(2.0 * jnp.pi * freq * t - phase))

    @property
    def name(self) -> str:
        return "time_domain_triangle"


class Pulse9(FourierPulseShape):
    def term(self, m, t, freq, phase):
        p = jnp.pi / jnp.sqrt(3.0)
        return ((-1) ** m / m) * jnp.cos(
            2.0 * jnp.pi * m * freq * t - m * phase - m * p
        )

    @property
    def norm_factor(self) -> float | jax.Array:
        return -jnp.sum(1.0 / jnp.arange(1, NUM_FOURIER_TERMS + 1))

    @property
    def name(self) -> str:
        return "pulse_9"


class Pulse10(FourierPulseShape):
    def term(self, m, t, freq, phase):
        return ((-1) ** m / m) * jnp.sin(
            2.0 * jnp.pi * (2 * m - 1) * freq * t - (2 * m - 1) * phase
        )

    @property
    def norm_factor(self) -> float | jax.Array:
        return jnp.sum(1.0 / jnp.arange(1, NUM_FOURIER_TERMS + 1))

    @property
    def name(self) -> str:
        return "pulse_10"


class Rect75(FourierPulseShape):
    """75% duty rectangular waveform (+1 for 25%, -1 for 75%)."""

    @property
    def duty(self):
        return 0.75

    def term(self, m, t, freq, phase):

        # Constants
        D = self.duty
        omega = 2.0 * jnp.pi * freq

        # Fourier cosine/sine coefficients
        # a_m = 2/(π m) * sin(2π m D)
        # b_m = 2/(π m) * (1 - cos(2π m D))
        a_m = (2.0 / (jnp.pi * m)) * jnp.sin(2.0 * jnp.pi * m * D)
        b_m = (2.0 / (jnp.pi * m)) * (1.0 - jnp.cos(2.0 * jnp.pi * m * D))

        angle = m * omega * t - m * phase
        return a_m * jnp.cos(angle) + b_m * jnp.sin(angle)

    @property
    def norm_factor(self):
        return 1.0

    @property
    def name(self):
        return "Rectangle 25% duty"

    def __call__(self, t, freq, phase, initial_time):
        t = t - initial_time

        # JAX array of harmonics
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1, dtype=float)

        # Vectorized evaluation of each term
        y = jnp.sum(jax.vmap(lambda k: self.term(k, t, freq, phase))(m), axis=0)

        # DC offset for duty‑cycle rectangular waves
        dc = 2.0 * (1 - self.duty) - 1.0  # = -0.5 at D=0.25

        return y + dc


class Rect25(FourierPulseShape):
    """25% duty rectangular waveform (+1 for 75%, -1 for 25%)."""

    @property
    def duty(self):
        return 0.25  # D = 3/4

    def term(self, m, t, freq, phase):

        # Constants
        D = self.duty
        omega = 2.0 * jnp.pi * freq

        # Fourier cosine/sine coefficients for general duty D
        # a_m = 2/(π m) * sin(2π m D)
        # b_m = 2/(π m) * (1 - cos(2π m D))
        a_m = (2.0 / (jnp.pi * m)) * jnp.sin(2.0 * jnp.pi * m * D)
        b_m = (2.0 / (jnp.pi * m)) * (1.0 - jnp.cos(2.0 * jnp.pi * m * D))

        angle = m * omega * t - m * phase
        return a_m * jnp.cos(angle) + b_m * jnp.sin(angle)

    @property
    def norm_factor(self):
        return 1.0

    @property
    def name(self):
        return "Rectangle 75% duty"

    def __call__(self, t, freq, phase, initial_time):
        t = t - initial_time

        # JAX array of harmonics
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1, dtype=float)

        # Vectorized evaluation of each term and sum
        y = jnp.sum(jax.vmap(lambda k: self.term(k, t, freq, phase))(m), axis=0)

        # DC offset for duty-cycle rectangular waves: DC = 2D - 1
        dc = 2.0 * (1 - self.duty) - 1.0  # = +0.5 at D=0.75

        return y + dc


class Mono99(FourierPulseShape):
    """
    Rectangular waveform:
      - First 99% of cycle:  -1
      - Last  1% of cycle:    0

    Implemented as a 0-level window of duty D=0.01, phase-shifted
    to occur at the end of each period.
    """

    @property
    def duty(self):
        # Fraction of the period at the high level A (here A = 0)
        return 0.01

    @property
    def high_level(self):
        # A
        return 0.0

    @property
    def low_level(self):
        # B
        return -1.0

    @property
    def phase_offset(self):
        # Fixed offset to place the 0-level window at the end (last 1% of the cycle).
        # Δφ = 2π * (1 - D) = 2π * 0.99 = 1.98π
        return 1.98 * jnp.pi

    def term(self, m, t, freq, phase):

        # Constants
        D = self.duty
        A = self.high_level
        B = self.low_level
        omega = 2.0 * jnp.pi * freq

        # Fourier coefficients for general A,B,D
        # a_m = (A-B)/(π m) * sin(2π m D)
        # b_m = (A-B)/(π m) * (1 - cos(2π m D))
        factor = (A - B) / (jnp.pi * m)  # here = 1/(π m)
        a_m = factor * jnp.sin(2.0 * jnp.pi * m * D)
        b_m = factor * (1.0 - jnp.cos(2.0 * jnp.pi * m * D))

        # Apply the fixed phase offset so the 0-level window is at the end
        angle = m * (omega * t - (phase + self.phase_offset))
        return a_m * jnp.cos(angle) + b_m * jnp.sin(angle)

    @property
    def norm_factor(self):
        return 1.0

    @property
    def name(self):
        return "Monopole 99%"

    def __call__(self, t, freq, phase, initial_time):
        # Shift time origin
        t = t - initial_time

        # Harmonic indices
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1, dtype=float)

        # Sum harmonic contributions
        y = jnp.sum(jax.vmap(lambda k: self.term(k, t, freq, phase))(m), axis=0)

        # DC component: DC = A*D + B*(1-D) = 0*0.01 + (-1)*0.99 = -0.99
        D = self.duty
        A = self.high_level
        B = self.low_level
        dc = A * D + B * (1.0 - D)

        return y + dc


class Mono95(FourierPulseShape):
    """
    Rectangular waveform:
      - First 95% of cycle:  -1
      - Last  5% of cycle:   0

    Implemented as a 0-level window of duty D = 0.05, phase-shifted
    to occur at the end of each period.
    """

    @property
    def duty(self):
        # D is the fraction spent at the "high" level A (here A = 0)
        return 0.05

    @property
    def high_level(self):
        # A
        return 0.0

    @property
    def low_level(self):
        # B
        return -1.0

    @property
    def phase_offset(self):
        # Shift the 0-level window to the last D fraction of the cycle.
        # Δφ = 2π * (1 - D) = 2π * 0.95 = 1.9π
        return 1.9 * jnp.pi

    def term(self, m, t, freq, phase):

        # Constants
        D = self.duty
        A = self.high_level
        B = self.low_level
        omega = 2.0 * jnp.pi * freq

        # General rectangular coefficients for (A for D, B for 1-D)
        # a_m = (A-B)/(π m) * sin(2π m D)
        # b_m = (A-B)/(π m) * (1 - cos(2π m D))
        factor = (A - B) / (jnp.pi * m)  # = 1/(π m)
        a_m = factor * jnp.sin(2.0 * jnp.pi * m * D)
        b_m = factor * (1.0 - jnp.cos(2.0 * jnp.pi * m * D))

        # Apply fixed phase offset so the 0 window is at the end of the period
        angle = m * (omega * t - (phase + self.phase_offset))
        return a_m * jnp.cos(angle) + b_m * jnp.sin(angle)

    @property
    def norm_factor(self):
        return 1.0

    @property
    def name(self):
        return "Monopole 95%"

    def __call__(self, t, freq, phase, initial_time):
        # Shift time origin
        t = t - initial_time

        # Harmonic indices
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1, dtype=float)

        # Sum harmonic contributions
        y = jnp.sum(jax.vmap(lambda k: self.term(k, t, freq, phase))(m), axis=0)

        # DC component: DC = A*D + B*(1-D) = 0*0.05 + (-1)*0.95 = -0.95
        D = self.duty
        A = self.high_level
        B = self.low_level
        dc = A * D + B * (1.0 - D)

        return y + dc


class Rect75NegPos(FourierPulseShape):
    """
    Rectangular waveform:
      - First 75% of cycle:  -1
      - Last  25% of cycle:  +1

    Implemented as a +1 window of duty D = 0.75, phase-shifted
    to occur at the end (last quarter) of each period.
    """

    @property
    def duty(self):
        # Fraction of the period at the +1 level
        return 0.75

    @property
    def high_level(self):
        # A
        return 1.0

    @property
    def low_level(self):
        # B
        return -1.0

    @property
    def phase_offset(self):
        # Shift to place the +1 window at the end: Δφ = 2π * (1 - D) = 1.5π
        return 1.5 * jnp.pi

    def term(self, m, t, freq, phase):

        # Constants
        D = 1 - self.duty
        A = self.high_level
        B = self.low_level
        omega = 2.0 * jnp.pi * freq

        # General rectangular coefficients for (A for D, B for 1-D)
        # a_m = (A-B)/(π m) * sin(2π m D)
        # b_m = (A-B)/(π m) * (1 - cos(2π m D))
        factor = (A - B) / (jnp.pi * m)  # here = 2/(π m)
        a_m = factor * jnp.sin(2.0 * jnp.pi * m * D)
        b_m = factor * (1.0 - jnp.cos(2.0 * jnp.pi * m * D))

        # Fixed phase offset so the +1 window is in the last quarter
        angle = m * (omega * t - (phase + self.phase_offset))
        return a_m * jnp.cos(angle) + b_m * jnp.sin(angle)

    @property
    def norm_factor(self):
        return 1.0

    @property
    def name(self):
        return "Rect25NegPos"

    def __call__(self, t, freq, phase, initial_time):
        # Shift time origin
        t = t - initial_time

        # Harmonic indices
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1, dtype=float)

        # Sum harmonic contributions
        y = jnp.sum(jax.vmap(lambda k: self.term(k, t, freq, phase))(m), axis=0)

        # DC component: DC = A*D + B*(1-D) = +1*0.25 + (-1)*0.75 = -0.5
        D = 1 - self.duty
        A = self.high_level
        B = self.low_level
        dc = A * D + B * (1.0 - D)

        return y + dc


class Rect25NegPos(FourierPulseShape):
    """
    Rectangular waveform:
      - First 25% of cycle:  -1
      - Last  75% of cycle:  +1

    Implemented as a +1 window of duty D = 0.25, phase-shifted so it begins
    at 25% into the cycle (i.e., occupies the last 75%).
    """

    @property
    def duty(self):
        # Fraction of the period at the +1 level
        return 0.25

    @property
    def high_level(self):
        # A
        return 1.0

    @property
    def low_level(self):
        # B
        return -1.0

    @property
    def phase_offset(self):
        # Shift to make the +1 window start at 25%: Δφ = 2π * (1 - D) = π/2
        return 0.5 * jnp.pi

    def term(self, m, t, freq, phase):

        # Constants
        D = 1 - self.duty
        A = self.high_level
        B = self.low_level
        omega = 2.0 * jnp.pi * freq

        # General rectangular coefficients for (A for D, B for 1-D)
        # a_m = (A-B)/(π m) * sin(2π m D)
        # b_m = (A-B)/(π m) * (1 - cos(2π m D))
        factor = (A - B) / (jnp.pi * m)  # = 2/(π m)
        a_m = factor * jnp.sin(2.0 * jnp.pi * m * D)
        b_m = factor * (1.0 - jnp.cos(2.0 * jnp.pi * m * D))

        # Fixed phase offset so the +1 window begins at 25% of each period
        angle = m * (omega * t - (phase + self.phase_offset))
        return a_m * jnp.cos(angle) + b_m * jnp.sin(angle)

    @property
    def norm_factor(self):
        return 1.0

    @property
    def name(self):
        return "Rect75NegPos"

    def __call__(self, t, freq, phase, initial_time):
        # Shift time origin
        t = t - initial_time

        # Harmonic indices
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1, dtype=float)

        # Sum harmonic contributions
        y = jnp.sum(jax.vmap(lambda k: self.term(k, t, freq, phase))(m), axis=0)

        # DC component: DC = A*D + B*(1-D) = +1*0.75 + (-1)*0.25 = +0.5
        D = 1 - self.duty
        A = self.high_level
        B = self.low_level
        dc = A * D + B * (1.0 - D)

        return y + dc


class SquareNegPos(FourierPulseShape):
    """
    Rectangular waveform:
      - First 50% of cycle:  -1
      - Second 50% of cycle: +1

    Implemented as a +1 window of duty D = 0.5, phase-shifted by π so that
    the +1 segment occupies the second half of each period.
    """

    @property
    def duty(self):
        # Fraction of the period at the +1 level
        return 0.5

    @property
    def high_level(self):
        # A
        return 1.0

    @property
    def low_level(self):
        # B
        return -1.0

    @property
    def phase_offset(self):
        # Half-period shift: Δφ = 2π * 0.5 = π
        return jnp.pi

    def term(self, m, t, freq, phase):

        # Constants
        D = 1 - self.duty
        A = self.high_level
        B = self.low_level
        omega = 2.0 * jnp.pi * freq

        # Fourier coefficients for a two-level rectangular
        # For D = 0.5 and A-B = 2:
        #   a_m = 0 for all m
        #   b_m = (2/(π m)) * (1 - cos(π m)) = 0 for even m, 4/(π m) for odd m
        # We'll use the general form for clarity/JIT-friendliness:
        factor = (A - B) / (jnp.pi * m)  # = 2/(π m)
        a_m = factor * jnp.sin(2.0 * jnp.pi * m * D)  # -> 0
        b_m = factor * (
            1.0 - jnp.cos(2.0 * jnp.pi * m * D)
        )  # -> 0 (even m), 4/(π m) (odd m)

        # Apply the fixed phase offset so +1 occupies the second half
        angle = m * (omega * t - (phase + self.phase_offset))
        return a_m * jnp.cos(angle) + b_m * jnp.sin(angle)

    @property
    def norm_factor(self):
        return 1.0

    @property
    def name(self):
        return "Square NegPos"

    def __call__(self, t, freq, phase, initial_time):
        # Shift time origin
        t = t - initial_time

        # Harmonic indices
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1, dtype=float)

        # Sum harmonic contributions
        y = jnp.sum(jax.vmap(lambda k: self.term(k, t, freq, phase))(m), axis=0)

        # DC = A*D + B*(1-D) = 1*0.5 + (-1)*0.5 = 0
        return y


class Rect95(FourierPulseShape):
    """
    Rectangular waveform:
      - For 0 < t <= 0.95T:   -1
      - For 0.95T < t < T:    +1
    With f(0) defined as 0 (which the Fourier series attains as the midpoint at the jump).

    Implemented as a +1 window of duty D = 0.05, phase-shifted to occur
    at the end of each period.
    """

    @property
    def duty(self):
        # Fraction of the period at the +1 level
        return 0.95

    @property
    def high_level(self):
        return 1.0  # A

    @property
    def low_level(self):
        return -1.0  # B

    @property
    def phase_offset(self):
        # Δφ = 2π * (1 - D) = 1.9π to place +1 in the last 5%
        return 1.9 * jnp.pi

    def term(self, m, t, freq, phase):
        # Ensure float inside JAX tracing

        D = 1 - self.duty
        A = self.high_level
        B = self.low_level
        omega = 2.0 * jnp.pi * freq

        # General two-level rectangular coefficients
        factor = (A - B) / (jnp.pi * m)  # = 2/(π m)
        a_m = factor * jnp.sin(2.0 * jnp.pi * m * D)
        b_m = factor * (1.0 - jnp.cos(2.0 * jnp.pi * m * D))

        # Apply the fixed phase offset (places +1 at the end)
        angle = m * (omega * t - (phase + self.phase_offset))
        return a_m * jnp.cos(angle) + b_m * jnp.sin(angle)

    @property
    def norm_factor(self):
        return 1.0

    @property
    def name(self):
        return "rect_pos_last05"

    def __call__(self, t, freq, phase, initial_time):
        # Shift time origin
        t = t - initial_time

        # Harmonics
        m = jnp.arange(1, NUM_FOURIER_TERMS + 1, dtype=float)

        # Sum harmonics
        y = jnp.sum(jax.vmap(lambda k: self.term(k, t, freq, phase))(m), axis=0)

        # DC = A*D + B*(1-D) = +1*0.05 + (-1)*0.95 = -0.90
        D = 1 - self.duty
        A = self.high_level
        B = self.low_level
        dc = A * D + B * (1.0 - D)

        return y + dc
