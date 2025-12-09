"""Library of differentiable-enough pulse shapes."""

import abc
import equinox as eqx
import jax
import jax.numpy as jnp

NUM_FOURIER_TERMS = 50

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


class Triangle(FourierPulseShape):

    def term(self, m, t, freq, phase):
        return -((1 - (-1) ** m) / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * (t + (1.0 / (4.0 * freq))) - m * phase
        )

    @property
    def norm_factor(self) -> float | jax.Array:
        return (jnp.pi ** 2) / 4.0

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
        return (jnp.pi ** 2) / 6.0

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
        return (1.0 / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * t - m * phase
        ) - (1.0 / m) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)

    @property
    def norm_factor(self) -> float | jax.Array:
        return -(jnp.pi ** 2) / 6.0 + jnp.pi / 2.0

    @property
    def name(self) -> str:
        return "asymmetrical"


class SlantedSine(FourierPulseShape):

    def term(self, m, t, freq, phase):
        return ((-1) ** m / (m**2)) * jnp.sin(
            2.0 * jnp.pi * m * freq * t - m * phase
        )

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

