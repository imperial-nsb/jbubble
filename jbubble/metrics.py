"""Common differentiable metrics for bubble dynamics.

All functions operate on plain `jax.Array` arguments and are fully
differentiable. They are building blocks for the `loss_fn` argument of
[`fit_parameters`][jbubble.fitting.fit_parameters], which receives a
[`SimulationResult`][jbubble.simulation.SimulationResult]. Extract the
field you want to fit before you pass it to these functions.

Examples
--------
```python
from jbubble.metrics import mse_emission, normalised_mse_radius

# fit on the radius waveform
loss_fn = lambda result: normalised_mse_radius(result.state.R, target, R0)

# fit on the acoustic emission
loss_fn = lambda result: mse_emission(
    emission_model(result, r_hydrophone), target_pressure
)
```
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "mse_radius",
    "normalised_mse_radius",
    "peak_expansion",
    "peak_expansion_error",
    "mse_emission",
    "normalised_mse_emission",
]


def mse_radius(r_sim: jax.Array, r_target: jax.Array) -> jax.Array:
    r"""Mean squared error between simulated and target radii [m²].

    $$
    \frac{1}{N}\sum_{i=1}^{N} \left(R_{\text{sim},i} - R_{\text{target},i}\right)^2
    $$

    Parameters
    ----------
    r_sim : jax.Array, shape (N,)
        Simulated radius trajectory [m].
    r_target : jax.Array, shape (N,)
        Target radius trajectory [m].

    Returns
    -------
    jax.Array
        Scalar mean squared error [m²].
    """
    return jnp.mean((r_sim - r_target) ** 2)


def normalised_mse_radius(
    r_sim: jax.Array,
    r_target: jax.Array,
    R0: ArrayLike,
) -> jax.Array:
    r"""Normalised mean squared radius error (dimensionless).

    $$
    \frac{1}{N}\sum_{i=1}^{N}
        \left(\frac{R_{\text{sim},i} - R_{\text{target},i}}{R_0}\right)^2
    $$

    Equivalent to `mse_radius(r_sim / R0, r_target / R0)`. The $R_0$
    normalisation makes the loss dimensionless and of order 1 regardless of
    bubble size, which improves optimiser conditioning.

    Parameters
    ----------
    r_sim : jax.Array, shape (N,)
        Simulated radius trajectory [m].
    r_target : jax.Array, shape (N,)
        Target radius trajectory [m].
    R0 : float or jax.Array
        Equilibrium radius used for normalisation [m].

    Returns
    -------
    jax.Array
        Scalar normalised mean squared error.
    """
    return jnp.mean(((r_sim - r_target) / R0) ** 2)


def peak_expansion(r_sim: jax.Array, R0: ArrayLike) -> jax.Array:
    r"""Maximum radial expansion ratio, $R_\text{max}/R_0$.

    Parameters
    ----------
    r_sim : jax.Array, shape (N,)
        Simulated radius trajectory [m].
    R0 : float or jax.Array
        Equilibrium radius [m].

    Returns
    -------
    jax.Array
        Scalar expansion ratio (dimensionless).
    """
    return jnp.max(r_sim) / R0


def peak_expansion_error(
    r_sim: jax.Array,
    R0: ArrayLike,
    target_expansion: ArrayLike,
) -> jax.Array:
    r"""Squared error in the peak expansion ratio.

    $$
    \left(\frac{R_\text{max}}{R_0} - \mathrm{target\_expansion}\right)^2
    $$

    Useful when only the maximum oscillation amplitude matters rather
    than the full waveform.

    Parameters
    ----------
    r_sim : jax.Array, shape (N,)
        Simulated radius trajectory [m].
    R0 : float or jax.Array
        Equilibrium radius [m].
    target_expansion : float or jax.Array
        Target $R_\text{max}/R_0$ value.

    Returns
    -------
    jax.Array
        Scalar squared error (dimensionless).
    """
    return (peak_expansion(r_sim, R0) - target_expansion) ** 2


def mse_emission(
    p_sim: jax.Array,
    p_target: jax.Array,
) -> jax.Array:
    """Mean squared error between simulated and target emission pressure [Pa²].

    Parameters
    ----------
    p_sim : jax.Array, shape (N,)
        Simulated radiated pressure [Pa].
    p_target : jax.Array, shape (N,)
        Target radiated pressure [Pa].

    Returns
    -------
    jax.Array
        Scalar mean squared error [Pa²].
    """
    return jnp.mean((p_sim - p_target) ** 2)


def normalised_mse_emission(
    p_sim: jax.Array,
    p_target: jax.Array,
    p_ref: ArrayLike,
) -> jax.Array:
    r"""Normalised mean squared error (MSE) for emission pressure (dimensionless).

    $$
    \frac{1}{N}\sum_{i=1}^{N}
        \left(\frac{p_{\text{sim},i} - p_{\text{target},i}}{p_\text{ref}}\right)^2
    $$

    Parameters
    ----------
    p_sim : jax.Array, shape (N,)
        Simulated radiated pressure [Pa].
    p_target : jax.Array, shape (N,)
        Target radiated pressure [Pa].
    p_ref : float or jax.Array
        Reference pressure for normalisation [Pa].

    Returns
    -------
    jax.Array
        Scalar normalised mean squared error.
    """
    return jnp.mean(((p_sim - p_target) / p_ref) ** 2)
