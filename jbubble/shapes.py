"""Library of differentiable-enough pulse shapes."""

import jax
import jax.numpy as jnp

NUM_FOURIER_TERMS = 50

# Analytical normalization factors (based on infinite series limits or coefficient sums)
NORM_SAWTOOTH = jnp.pi / 2.0
NORM_TRIANGLE = (jnp.pi ** 2) / 4.0
NORM_QUADRATIC = (jnp.pi ** 2) / 6.0
NORM_ASYMMETRICAL = (jnp.pi ** 2) / 6.0 + jnp.pi / 2.0
NORM_SLANTED_SINE = 1.0149  # Approx max of Clausen function Cl2
NORM_SQUARE = jnp.pi / 4.0

# For divergent series, we normalize by the sum of absolute coefficients
_harmonic_sum = jnp.sum(1.0 / jnp.arange(1, NUM_FOURIER_TERMS + 1))
NORM_PULSE_9 = _harmonic_sum
NORM_PULSE_10 = _harmonic_sum


def pulse_sine(t, freq, phase, initial_time):
    """Pure sine wave."""
    t = t - initial_time
    return jnp.sin(2.0 * jnp.pi * freq * t - phase)


def pulse_sawtooth(t, freq, phase, initial_time):
    """Fourier sawtooth approximation."""
    t = t - initial_time
    m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

    def term_fn(m_val):
        return -((-1) ** m_val / m_val) * jnp.sin(
            2.0 * jnp.pi * m_val * freq * t - m_val * phase
        )

    return jnp.sum(jax.vmap(term_fn)(m), axis=0) / NORM_SAWTOOTH


def pulse_triangle(t, freq, phase, initial_time):
    t = t - initial_time
    m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

    def term_fn(m_val):
        return -((1 - (-1) ** m_val) / (m_val**2)) * jnp.cos(
            2.0 * jnp.pi * m_val * freq * (t + (1.0 / (4.0 * freq))) - m_val * phase
        )

    return jnp.sum(jax.vmap(term_fn)(m), axis=0) / NORM_TRIANGLE


def pulse_quadratic(t, freq, phase, initial_time):
    t = t - initial_time
    p = jnp.pi / jnp.sqrt(3.0)
    m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

    def term_fn(m_val):
        return ((-1) ** m_val / (m_val**2)) * jnp.cos(
            2.0 * jnp.pi * m_val * freq * t - m_val * phase - m_val * p
        )

    return jnp.sum(jax.vmap(term_fn)(m), axis=0) / NORM_QUADRATIC


def pulse_negative_quadratic(t, freq, phase, initial_time):
    t = t - initial_time
    p = jnp.pi / jnp.sqrt(3.0)
    m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

    def term_fn(m_val):
        return -(((-1) ** m_val) / (m_val**2)) * jnp.cos(
            2.0 * jnp.pi * m_val * freq * t - m_val * phase - m_val * p
        )

    return jnp.sum(jax.vmap(term_fn)(m), axis=0) / NORM_QUADRATIC


def pulse_asymmetrical(t, freq, phase, initial_time):
    t = t - initial_time
    m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

    def term_fn(m_val):
        return (1.0 / (m_val**2)) * jnp.cos(
            2.0 * jnp.pi * m_val * freq * t - m_val * phase
        ) - (1.0 / m_val) * jnp.sin(2.0 * jnp.pi * m_val * freq * t - m_val * phase)

    return -jnp.sum(jax.vmap(term_fn)(m), axis=0) / NORM_ASYMMETRICAL


def pulse_slanted_sine(t, freq, phase, initial_time):
    t = t - initial_time
    m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

    def term_fn(m_val):
        return ((-1) ** m_val / (m_val**2)) * jnp.sin(
            2.0 * jnp.pi * m_val * freq * t - m_val * phase
        )

    return jnp.sum(jax.vmap(term_fn)(m), axis=0) / NORM_SLANTED_SINE


def pulse_square(t, freq, phase, initial_time):
    t = t - initial_time
    m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

    def term_fn(m_val):
        return (1.0 / (2 * m_val - 1)) * jnp.sin(
            2.0 * jnp.pi * (2 * m_val - 1) * freq * t - (2 * m_val - 1) * phase
        )

    return jnp.sum(jax.vmap(term_fn)(m), axis=0) / NORM_SQUARE


def pulse_9(t, freq, phase, initial_time):
    t = t - initial_time
    p = jnp.pi / jnp.sqrt(3.0)
    m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

    def term_fn(m_val):
        return ((-1) ** m_val / m_val) * jnp.cos(
            2.0 * jnp.pi * m_val * freq * t - m_val * phase - m_val * p
        )

    return -jnp.sum(jax.vmap(term_fn)(m), axis=0) / NORM_PULSE_9


def pulse_10(t, freq, phase, initial_time):
    t = t - initial_time
    m = jnp.arange(1, NUM_FOURIER_TERMS + 1)

    def term_fn(m_val):
        return ((-1) ** m_val / m_val) * jnp.sin(
            2.0 * jnp.pi * (2 * m_val - 1) * freq * t - (2 * m_val - 1) * phase
        )

    return jnp.sum(jax.vmap(term_fn)(m), axis=0) / NORM_PULSE_10


DEFAULT_PULSE_LIBRARY = {
    "sine": pulse_sine,
    "sawtooth": pulse_sawtooth,
    "triangle": pulse_triangle,
    "quadratic": pulse_quadratic,
    "neg_quadratic": pulse_negative_quadratic,
    "asym": pulse_asymmetrical,
    "slanted_sine": pulse_slanted_sine,
    "square": pulse_square,
    "9": pulse_9,
    "10": pulse_10,
}
