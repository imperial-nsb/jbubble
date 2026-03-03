"""
Shared pressure term functions used across bubble models.

These functions encapsulate the standard formulas for gas pressure, Laplace
pressure, viscous damping, and shell viscosity that appear identically
across multiple bubble model implementations.
"""


def gas_pressure_equilibrium(P_amb, sigma_R0, R0):
    """Equilibrium gas pressure: P₀ = P_amb + 2σ(R0)/R0.

    Parameters
    ----------
    P_amb : float
        Ambient pressure.
    sigma_R0 : float
        Surface tension at equilibrium.
    R0 : float
        Equilibrium radius.

    Returns
    -------
    float
        Equilibrium gas pressure.
    """
    return P_amb + 2.0 * sigma_R0 / R0


def gas_pressure(P_gas0, R0, R, gamma):
    """Polytropic gas pressure: P = P₀ (R₀/R)^(3γ).

    Parameters
    ----------
    P_gas0 : float or array
        Equilibrium gas pressure.
    R0 : float
        Equilibrium radius.
    R : float or array
        Current radius.
    gamma : float
        Polytropic exponent.

    Returns
    -------
    float or array
        Gas pressure at radius *R*.
    """
    return P_gas0 * (R0 / R) ** (3 * gamma)


def laplace_pressure(sigma, R):
    """Laplace pressure: P_L = 2σ/R.

    Parameters
    ----------
    sigma : float or array
        Surface tension.
    R : float or array
        Bubble radius.

    Returns
    -------
    float or array
        Laplace pressure.
    """
    return 2.0 * sigma / R


def viscous_pressure(mu, R_dot, R):
    """Liquid viscous pressure: P_μ = 4μṘ/R.

    Parameters
    ----------
    mu : float
        Liquid dynamic viscosity.
    R_dot : float or array
        Radius velocity.
    R : float or array
        Bubble radius.

    Returns
    -------
    float or array
        Viscous pressure.
    """
    return 4.0 * mu * R_dot / R


def shell_viscous_pressure(kappa_s, R_dot, R):
    """Shell surface-dilatational viscous pressure: P_κ = 4κ_s Ṙ/R².

    Parameters
    ----------
    kappa_s : float
        Shell surface-dilatational viscosity.
    R_dot : float or array
        Radius velocity.
    R : float or array
        Bubble radius.

    Returns
    -------
    float or array
        Shell viscous pressure.
    """
    return 4.0 * kappa_s * R_dot / R**2
