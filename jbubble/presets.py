"""Single source of truth for physical constants and ready-to-use bubble models.

This module provides standard material parameters and pre-configured
EquationOfMotion components via functools.partial and model factories.
"""

from functools import partial
from typing import Type

import jax.numpy as jnp

from .bubble.eom import EquationOfMotion, KellerMiksis
from .bubble.gas import PolytropicGas, VanDerWaalsGas
from .bubble.medium import NewtonianMedium, KelvinVoigtMedium, NeoHookeanMedium
from .bubble.properties import ConstantProperty, GompertzSurfaceTension, Property
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

VANDERWAALS_HDIV = 5.61   # unitless

# =====================================================================
# B. Generic Gas Constructors
# =====================================================================

def calculate_equilibrium_pgas(R0: float, P_amb: float, sigma: Property) -> float:
    """Compute the equilibrium gas pressure via Laplace balance.
    
    P_gas0 = P_amb + 2 sigma(R0) / R0.
    """
    state_0 = BubbleState(R=jnp.array(R0), R_dot=jnp.array(0.0))
    sigma_r0_val = sigma(state_0)
    return float(P_amb + 2.0 * sigma_r0_val / R0)

def make_polytropic_gas(
    R0: float, gamma: float, P_amb: float, sigma: Property
) -> PolytropicGas:
    """Construct a PolytropicGas using physical equilibrium conditions."""
    P_gas0 = calculate_equilibrium_pgas(R0, P_amb, sigma)
    return PolytropicGas(P_gas0=P_gas0, R0=R0, gamma=gamma)

def make_vanderwaals_gas(
    R0: float, gamma: float, P_amb: float, sigma: Property, h_divisor: float = VANDERWAALS_HDIV
) -> VanDerWaalsGas:
    """Construct a VanDerWaalsGas using physical equilibrium conditions."""
    P_gas0 = calculate_equilibrium_pgas(R0, P_amb, sigma)
    h = R0 / h_divisor
    return VanDerWaalsGas(P_gas0=P_gas0, R0=R0, gamma=gamma, h=h)


# =====================================================================
# C. Functional Partials for Components
# =====================================================================

NewtonianWater = partial(NewtonianMedium, mu=WATER_MU)
KelvinVoigtWater = partial(KelvinVoigtMedium, mu=WATER_MU)
NeoHookeanWater = partial(NeoHookeanMedium, mu=WATER_MU)

PolytropicAir = partial(make_polytropic_gas, gamma=AIR_GAMMA)
PolytropicSF6 = partial(make_polytropic_gas, gamma=SF6_GAMMA)


# =====================================================================
# D. High-Level Macro Presets
# =====================================================================

def build_sono_vue(
    R0: float,
    eom_class: Type[EquationOfMotion] = KellerMiksis,
    P_amb: float = 101325.0,
) -> EquationOfMotion:
    """Assembles a standard SonoVue lipid microbubble in water."""
    sigma = GompertzSurfaceTension.from_R0(
        R0=R0, 
        R_buckle_ratio=0.99,
        chi=LIPID_CHI, 
        sigma_break=WATER_SIGMA
    )
    
    shell = LipidShell(
        sigma=sigma, 
        kappa_s=ConstantProperty(val=LIPID_KAPPA_S, _scale="kappa_scale")
    )
    
    gas = PolytropicSF6(R0=R0, P_amb=P_amb, sigma=sigma)
    medium = NewtonianWater()
    
    kwargs = {
        "gas": gas, 
        "shell": shell, 
        "medium": medium, 
        "R0": R0, 
        "P_amb": P_amb, 
        "rho_L": WATER_RHO
    }
    
    if hasattr(eom_class, "c_L"):
        kwargs["c_L"] = WATER_C
        
    return eom_class(**kwargs)
