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
    ChurchGompertz,
    KellerMiksisGompertz,
    KelvinVoigtGompertz,
    LeightonGompertz,
    Marmottant,
    MarmottantGompertz,
    NeoHookeanGompertz,
    RayleighPlesset,
    SphericalConfinement,
)

# All 9 concrete bubble model classes, used for parametrised tests
ALL_BUBBLE_CLASSES = [
    RayleighPlesset,
    Marmottant,
    MarmottantGompertz,
    KellerMiksisGompertz,
    KelvinVoigtGompertz,
    NeoHookeanGompertz,
    ChurchGompertz,
    LeightonGompertz,
    SphericalConfinement,
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
def rp_bubble():
    return RayleighPlesset(R0=3e-6)


@pytest.fixture(scope="session")
def marmottant_bubble():
    return Marmottant(R0=4e-6)
