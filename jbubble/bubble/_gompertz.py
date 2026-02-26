"""
Shared Gompertz surface-tension helper.

The Gompertz function provides a smooth, differentiable approximation to the
piecewise Marmottant surface tension law, enabling automatic differentiation.

    σ(R) = a · exp(−b · exp(c · (1 − R/R_buckle)))

where a, b, c are derived from the shell parameters so that:
  - σ → 0        as R → −∞  (buckled limit)
  - σ = σ_R0     at R = R0  (equilibrium)
  - σ → σ_break  as R → +∞  (ruptured limit)
"""

import jax.numpy as jnp


def gompertz_surface_tension(
    R,
    *,
    R0: float,
    R_buckle: float,
    chi: float,
    sigma_break: float,
    sigma_R0: float,
):
    """
    Evaluate the Gompertz surface tension at radius *R*.

    Parameters
    ----------
    R : scalar or array
        Bubble radius.
    R0 : float
        Equilibrium radius.
    R_buckle : float
        Buckling radius (transition from buckled to elastic regime).
    chi : float
        Shell elasticity modulus [N/m].
    sigma_break : float
        Surface tension at shell rupture [N/m].
    sigma_R0 : float
        Surface tension at equilibrium (pre-computed as χ(R0²/R_buckle² − 1)).

    Returns
    -------
    Same type as *R*.
    """
    a = sigma_break
    c = (2.0 * chi / a) * jnp.sqrt(1.0 + a / (2.0 * chi))
    b = -jnp.log(sigma_R0 / a) / jnp.exp(c * (1.0 - R0 / R_buckle))
    return a * jnp.exp(-b * jnp.exp(c * (1.0 - R / R_buckle)))
