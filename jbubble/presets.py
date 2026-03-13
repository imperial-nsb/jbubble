"""Single source of truth for physical constants and ready-to-use bubble models."""

import dataclasses

import jax.numpy as jnp

from .bubble.eom import EquationOfMotion, KellerMiksis
from .bubble.gas import PolytropicGas
from .bubble.medium import NewtonianMedium
from .bubble.properties import GompertzSurfaceTension, Property
from .bubble.shell import LipidShell
from .bubble.state import BubbleState

# =====================================================================
# A. Material Constants
# =====================================================================

WATER_MU = 0.00089        # Pa s
WATER_RHO = 998.0         # kg/m^3
WATER_C = 1481.0          # m/s
WATER_SIGMA = 72e-3       # N/m

AIR_GAMMA = 1.4           # unitless
SF6_GAMMA = 1.07          # unitless

LIPID_CHI = 0.38          # N/m
LIPID_KAPPA_S = 2.4e-9    # N s/m

CHURCH_DS = 4e-9          # m
CHURCH_GS = 10e6          # Pa
CHURCH_MUS = 0.5          # Pa s


# =====================================================================
# B. Utility
# =====================================================================

def calculate_equilibrium_pgas(R0: float, P_amb: float, sigma: Property) -> float:
    """Compute the equilibrium gas pressure via Laplace balance.

    P_gas0 = P_amb + 2 sigma(R0) / R0.
    """
    state_0 = BubbleState(
        R=jnp.array(R0), R_dot=jnp.array(0.0),
        R0=jnp.array(R0), P_gas0=jnp.array(0.0),
    )
    sigma_r0_val = sigma(state_0)
    return float(P_amb + 2.0 * sigma_r0_val / R0)


# =====================================================================
# C. High-Level Macro Presets
# =====================================================================

def build_sono_vue(
    R0: float,
    eom_class: type[EquationOfMotion] = KellerMiksis,
    P_amb: float = 101325.0,
) -> EquationOfMotion:
    """Assemble a standard SonoVue lipid microbubble in water."""
    sigma = GompertzSurfaceTension.from_R0(
        R0=R0,
        R_buckle_ratio=0.99,
        chi=LIPID_CHI,
        sigma_break=WATER_SIGMA,
    )
    shell  = LipidShell(sigma=sigma, kappa_s=LIPID_KAPPA_S)
    gas    = PolytropicGas(gamma=SF6_GAMMA)
    medium = NewtonianMedium(mu=WATER_MU)

    kwargs: dict = dict(
        gas=gas, shell=shell, medium=medium,
        R0=R0, P_amb=P_amb, rho_L=WATER_RHO,
    )
    if "c_L" in {f.name for f in dataclasses.fields(eom_class)}:
        kwargs["c_L"] = WATER_C

    return eom_class(**kwargs)
