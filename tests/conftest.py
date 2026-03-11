"""Shared fixtures and configuration for the jbubble test suite.

IMPORTANT: jax_enable_x64 must be set before any JAX tracing occurs.
conftest.py is loaded by pytest before any test modules are imported,
so this is the correct place to configure JAX.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jbubble.shapes as shapes  # noqa: E402
import pytest  # noqa: E402
from jbubble import Pulse, SaveSpec, Units  # noqa: E402
from jbubble.bubble import (  # noqa: E402
    ConstantSigma,
    GompertzSigma,
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
    """RayleighPlesset with uncoated bubble (ConstantSigma + NoShell)."""
    sigma = ConstantSigma()
    shell = NoShell(sigma=sigma)
    gas = PolytropicGas.from_equilibrium(
        R0=R0,
        gamma=1.4,
        P_amb=101325.0,
        sigma_R0=sigma.sigma_L,
    )
    medium = KelvinVoigtMedium(R0=R0)
    return RayleighPlesset(
        gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=998.0
    )


def make_gompertz_lipid(eom_cls, R0, **extra):
    """Compose an EoM with Gompertz + LipidShell + KelvinVoigtMedium."""
    sigma = GompertzSigma.from_R0(R0=R0)
    shell = LipidShell(sigma=sigma)
    gas = PolytropicGas.from_equilibrium(
        R0=R0,
        gamma=1.07,
        P_amb=101325.0,
        sigma_R0=sigma.sigma_R0,
    )
    medium = KelvinVoigtMedium(R0=R0)
    return eom_cls(
        gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=998.0, **extra
    )


def make_neo_hookean(R0):
    """KellerMiksis with NeoHookeanMedium."""
    sigma = GompertzSigma.from_R0(R0=R0)
    shell = LipidShell(sigma=sigma)
    gas = PolytropicGas.from_equilibrium(
        R0=R0,
        gamma=1.07,
        P_amb=101325.0,
        sigma_R0=sigma.sigma_R0,
    )
    medium = NeoHookeanMedium(R0=R0)
    return KellerMiksis(
        gas=gas,
        shell=shell,
        medium=medium,
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1481.0,
    )


def make_thick_shell(R0):
    """RayleighPlesset with ThickShell (Church model)."""
    sigma = GompertzSigma.from_R0(R0=R0)
    shell = ThickShell(sigma=sigma, R0=R0)
    gas = PolytropicGas.from_equilibrium(
        R0=R0,
        gamma=1.07,
        P_amb=101325.0,
        sigma_R0=sigma.sigma_R0,
    )
    medium = KelvinVoigtMedium(R0=R0)
    return RayleighPlesset(
        gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=998.0
    )


def make_confinement(R0):
    """SphericalConfinement with default vessel parameters."""
    sigma = GompertzSigma.from_R0(R0=R0)
    shell = LipidShell(sigma=sigma)
    gas = PolytropicGas.from_equilibrium(
        R0=R0,
        gamma=1.07,
        P_amb=101325.0,
        sigma_R0=sigma.sigma_R0,
    )
    medium = KelvinVoigtMedium(R0=R0)
    return SphericalConfinement(
        gas=gas,
        shell=shell,
        medium=medium,
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1481.0,
        vessel_radius=100e-6,
        vessel_rho=1100.0,
        vessel_E=1e6,
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
def units():
    return Units()


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
