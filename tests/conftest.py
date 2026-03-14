"""Shared fixtures and configuration for the jbubble test suite.

IMPORTANT: jax_enable_x64 must be set before any JAX tracing occurs.
conftest.py is loaded by pytest before any test modules are imported,
so this is the correct place to configure JAX.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jbubble.shapes as shapes  # noqa: E402
import pytest  # noqa: E402
from jbubble import Pulse, SaveSpec  # noqa: E402
from jbubble.bubble import (  # noqa: E402
    ConstantProperty,
    GompertzSurfaceTension,
    KellerMiksis,
    KelvinVoigtMedium,
    LeightonTube,
    LipidShell,
    ModifiedRayleighPlesset,
    NeoHookeanMedium,
    NoShell,
    PolytropicGas,
    RayleighPlesset,
    SphericalConfinement,
    ThickShell,
)

# ---------------------------------------------------------------------------
# Helpers: compose an EoM from just R0 (mirrors old monolithic constructors)
# ---------------------------------------------------------------------------


def make_rp(R0):
    """RayleighPlesset with uncoated bubble (ConstantProperty sigma + NoShell)."""
    from jbubble.presets import WATER_MU, WATER_RHO, AIR_GAMMA, make_polytropic_gas
    sigma = ConstantProperty(val=72e-3)
    shell = NoShell(sigma=sigma)
    gas = make_polytropic_gas(
        R0=R0,
        gamma=AIR_GAMMA,
        P_amb=101325.0,
        sigma=sigma,
    )
    medium = KelvinVoigtMedium(R0=R0, G=0.0, mu=WATER_MU)
    return RayleighPlesset(
        gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=WATER_RHO
    )


def make_gompertz_lipid(eom_cls, R0, **extra):
    """Compose an EoM with Gompertz + LipidShell + KelvinVoigtMedium."""
    from jbubble.presets import WATER_MU, WATER_RHO, LIPID_CHI, WATER_SIGMA, LIPID_KAPPA_S, SF6_GAMMA, make_polytropic_gas
    sigma = GompertzSurfaceTension.from_R0(
        R0=R0, R_buckle_ratio=0.99, chi=LIPID_CHI, sigma_break=WATER_SIGMA
    )
    shell = LipidShell(
        sigma=sigma,
        kappa_s=ConstantProperty(val=LIPID_KAPPA_S),
    )
    gas = make_polytropic_gas(
        R0=R0,
        gamma=SF6_GAMMA,
        P_amb=101325.0,
        sigma=sigma,
    )
    medium = KelvinVoigtMedium(R0=R0, G=0.0, mu=WATER_MU)
    return eom_cls(
        gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=WATER_RHO, **extra
    )


def make_neo_hookean(R0):
    """KellerMiksis with NeoHookeanMedium."""
    from jbubble.presets import WATER_MU, WATER_RHO, LIPID_CHI, WATER_SIGMA, LIPID_KAPPA_S, WATER_C, SF6_GAMMA, make_polytropic_gas
    sigma = GompertzSurfaceTension.from_R0(
        R0=R0, R_buckle_ratio=0.99, chi=LIPID_CHI, sigma_break=WATER_SIGMA
    )
    shell = LipidShell(
        sigma=sigma,
        kappa_s=ConstantProperty(val=LIPID_KAPPA_S),
    )
    gas = make_polytropic_gas(
        R0=R0,
        gamma=SF6_GAMMA,
        P_amb=101325.0,
        sigma=sigma,
    )
    medium = NeoHookeanMedium(R0=R0, G=10e3, mu=WATER_MU)
    return KellerMiksis(
        gas=gas,
        shell=shell,
        medium=medium,
        R0=R0,
        P_amb=101325.0,
        rho_L=WATER_RHO,
        c_L=WATER_C,
    )


def make_thick_shell(R0):
    """RayleighPlesset with ThickShell (Church model)."""
    from jbubble.presets import WATER_MU, WATER_RHO, WATER_SIGMA, LIPID_CHI, SF6_GAMMA, make_polytropic_gas, CHURCH_DS, CHURCH_GS, CHURCH_MUS
    sigma = GompertzSurfaceTension.from_R0(
        R0=R0, R_buckle_ratio=0.99, chi=LIPID_CHI, sigma_break=WATER_SIGMA
    )
    shell = ThickShell(sigma=sigma, R0=R0, d_s=CHURCH_DS, G_s=CHURCH_GS, mu_s=CHURCH_MUS)
    gas = make_polytropic_gas(
        R0=R0,
        gamma=SF6_GAMMA,
        P_amb=101325.0,
        sigma=sigma,
    )
    medium = KelvinVoigtMedium(R0=R0, G=0.0, mu=WATER_MU)
    return RayleighPlesset(
        gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=WATER_RHO
    )


def make_confinement(R0):
    """SphericalConfinement with default vessel parameters."""
    from jbubble.presets import WATER_MU, WATER_RHO, LIPID_CHI, WATER_SIGMA, LIPID_KAPPA_S, WATER_C, SF6_GAMMA, make_polytropic_gas
    sigma = GompertzSurfaceTension.from_R0(
        R0=R0, R_buckle_ratio=0.99, chi=LIPID_CHI, sigma_break=WATER_SIGMA
    )
    shell = LipidShell(
        sigma=sigma,
        kappa_s=ConstantProperty(val=LIPID_KAPPA_S),
    )
    gas = make_polytropic_gas(
        R0=R0,
        gamma=SF6_GAMMA,
        P_amb=101325.0,
        sigma=sigma,
    )
    medium = KelvinVoigtMedium(R0=R0, G=0.0, mu=WATER_MU)
    return SphericalConfinement(
        gas=gas,
        shell=shell,
        medium=medium,
        R0=R0,
        P_amb=101325.0,
        rho_L=WATER_RHO,
        c_L=WATER_C,
        vessel_radius=100e-6,
        vessel_rho=1100.0,
        vessel_E=1e6,
        vessel_nu=0.5,
        vessel_d=5e-6,
        tissue_rho=1000.0,
        tissue_d=20e-6,
    )


# All EoM factory functions, used for parametrised tests
ALL_EOM_FACTORIES = [
    pytest.param(lambda R0: make_rp(R0), id="RayleighPlesset"),
    pytest.param(lambda R0: make_gompertz_lipid(RayleighPlesset, R0), id="RP-Gompertz"),
    pytest.param(
        lambda R0: make_gompertz_lipid(ModifiedRayleighPlesset, R0, c_L=1481.0),
        id="ModifiedRP",
    ),
    pytest.param(
        lambda R0: make_gompertz_lipid(KellerMiksis, R0, c_L=1481.0), id="KellerMiksis"
    ),
    pytest.param(lambda R0: make_neo_hookean(R0), id="NeoHookean"),
    pytest.param(lambda R0: make_thick_shell(R0), id="ThickShell"),
    pytest.param(
        lambda R0: make_gompertz_lipid(
            LeightonTube, R0, c_L=1481.0, tube_radius=50e-6, tube_length=200e-6
        ),
        id="LeightonTube",
    ),
    pytest.param(lambda R0: make_confinement(R0), id="SphericalConfinement"),
]


@pytest.fixture(scope="session")
def pulse():
    return Pulse(
        freq=300e3,
        pressure=50e3,
        shape=shapes.Sine(),
        cycle_num=4,
        initial_time=1e-6,
        apply_hann=False,
    )


@pytest.fixture(scope="session")
def save_spec():
    # Small sample count so JIT compiles quickly in CI
    return SaveSpec(num_samples=64)


@pytest.fixture(scope="session")
def rp_eom():
    return make_rp(R0=3e-6)


@pytest.fixture(scope="session")
def km_eom():
    return make_gompertz_lipid(KellerMiksis, R0=4e-6, c_L=1481.0)
