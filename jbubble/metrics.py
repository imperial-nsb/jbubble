"""Common differentiable metrics for bubble dynamics fitting.

All functions operate on plain :class:`jax.Array` arguments and are
fully differentiable, so they can be passed directly as ``loss_fn`` to
:func:`~jbubble.fitting.fit_parameters` or composed into custom losses.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def mse_radius(r_sim: jax.Array, r_target: jax.Array) -> jax.Array:
    """Mean squared error between simulated and target radii [m²].

    Parameters
    ----------
    r_sim : jax.Array, shape (N,)
        Simulated radius trajectory [m].
    r_target : jax.Array, shape (N,)
        Target radius trajectory [m].
    """
    return jnp.mean((r_sim - r_target) ** 2)


def normalised_mse_radius(
    r_sim: jax.Array,
    r_target: jax.Array,
    R0: float | jax.Array,
) -> jax.Array:
    """Normalised mean squared radius error (dimensionless).

    Equivalent to ``mse_radius(r_sim / R0, r_target / R0)``.  The R0
    normalisation makes the loss dimensionless and ~O(1) regardless of
    bubble size, which improves optimiser conditioning.

    Parameters
    ----------
    r_sim : jax.Array, shape (N,)
        Simulated radius trajectory [m].
    r_target : jax.Array, shape (N,)
        Target radius trajectory [m].
    R0 : float or jax.Array
        Equilibrium radius used for normalisation [m].
    """
    return jnp.mean(((r_sim - r_target) / R0) ** 2)


def peak_expansion(r_sim: jax.Array, R0: float | jax.Array) -> jax.Array:
    """Maximum radial expansion ratio R_max / R0.

    Parameters
    ----------
    r_sim : jax.Array, shape (N,)
        Simulated radius trajectory [m].
    R0 : float or jax.Array
        Equilibrium radius [m].
    """
    return jnp.max(r_sim) / R0


def peak_expansion_error(
    r_sim: jax.Array,
    R0: float | jax.Array,
    target_expansion: float | jax.Array,
) -> jax.Array:
    """Squared error in peak expansion ratio.

    Useful when only the maximum oscillation amplitude matters rather
    than the full waveform.

    Parameters
    ----------
    r_sim : jax.Array, shape (N,)
        Simulated radius trajectory [m].
    R0 : float or jax.Array
        Equilibrium radius [m].
    target_expansion : float or jax.Array
        Target R_max / R0 value.
    """
    return (peak_expansion(r_sim, R0) - target_expansion) ** 2
