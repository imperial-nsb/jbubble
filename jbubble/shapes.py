"""Library of differentiable-enough pulse shapes."""

import jax.numpy as jnp


def pulse_sine(t, freq, num_fourier_terms, phase, initial_time):
    """Pure sine wave."""
    t = t - initial_time
    return jnp.sin(2.0 * jnp.pi * freq * t - phase)


def pulse_sawtooth(t, freq, num_fourier_terms, phase, initial_time):
    """Fourier sawtooth approximation."""
    t = t - initial_time
    output = 0.0
    for m in range(1, num_fourier_terms + 1):
        output += -((-1) ** m / m) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)
    return output


def pulse_triangle(t, freq, num_fourier_terms, phase, initial_time):
    t = t - initial_time
    output = 0.0
    for m in range(1, num_fourier_terms + 1):
        output += -((1 - (-1) ** m) / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * (t + (1.0 / (4.0 * freq))) - m * phase
        )
    return output


def pulse_quadratic(t, freq, num_fourier_terms, phase, initial_time):
    t = t - initial_time
    output = 0.0
    p = jnp.pi / jnp.sqrt(3.0)
    for m in range(1, num_fourier_terms + 1):
        output += ((-1) ** m / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * t - m * phase - m * p
        )
    return output


def pulse_negative_quadratic(t, freq, num_fourier_terms, phase, initial_time):
    t = t - initial_time
    output = 0.0
    p = jnp.pi / jnp.sqrt(3.0)
    for m in range(1, num_fourier_terms + 1):
        output += -(((-1) ** m) / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * t - m * phase - m * p
        )
    return output


def pulse_asymmetrical(t, freq, num_fourier_terms, phase, initial_time):
    t = t - initial_time
    output = 0.0
    for m in range(1, num_fourier_terms + 1):
        output += (1.0 / (m**2)) * jnp.cos(
            2.0 * jnp.pi * m * freq * t - m * phase
        ) - (1.0 / m) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)
    return -output


def pulse_slanted_sine(t, freq, num_fourier_terms, phase, initial_time):
    t = t - initial_time
    output = 0.0
    for m in range(1, num_fourier_terms + 1):
        output += ((-1) ** m / (m**2)) * jnp.sin(2.0 * jnp.pi * m * freq * t - m * phase)
    return output


def pulse_square(t, freq, num_fourier_terms, phase, initial_time):
    t = t - initial_time
    output = 0.0
    for m in range(1, num_fourier_terms + 1):
        output += (1.0 / (2 * m - 1)) * jnp.sin(
            2.0 * jnp.pi * (2 * m - 1) * freq * t - (2 * m - 1) * phase
        )
    return output


def pulse_9(t, freq, num_fourier_terms, phase, initial_time):
    t = t - initial_time
    output = 0.0
    p = jnp.pi / jnp.sqrt(3.0)
    for m in range(1, num_fourier_terms + 1):
        output += ((-1) ** m / m) * jnp.cos(
            2.0 * jnp.pi * m * freq * t - m * phase - m * p
        )
    return -output


def pulse_10(t, freq, num_fourier_terms, phase, initial_time):
    t = t - initial_time
    output = 0.0
    for m in range(1, num_fourier_terms + 1):
        output += ((-1) ** m / m) * jnp.sin(
            2.0 * jnp.pi * (2 * m - 1) * freq * t - (2 * m - 1) * phase
        )
    return output


DEFAULT_PULSE_LIBRARY = {
    "sine": pulse_sine,
    "sawtooth": pulse_sawtooth,
    "triangle": pulse_triangle,
    "quadratic": pulse_quadratic,
    "neg_quadratic": pulse_negative_quadratic,
    "asym": pulse_asymmetrical,
    "slanted_sine": pulse_slanted_sine,
    "square": pulse_square,
}
