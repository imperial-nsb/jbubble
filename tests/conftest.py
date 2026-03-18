"""Shared fixtures for jbubble tests."""

import jax.numpy as jnp
import pytest
from jbubble.bubble.eom import KellerMiksis, RayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.bubble.state import BubbleState
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine

# ── Physical constants ──────────────────────────────────────────────────────

R0 = 2e-6
P_AMB = 101325.0
RHO_L = 998.0
C_L = 1500.0
SIGMA = 0.072
MU = 1e-3
GAMMA = 1.4

FREQ = 1e6
PRESSURE = 100e3


@pytest.fixture
def equilibrium_state():
    """BubbleState at equilibrium: R = R0, R_dot = 0."""
    R0_arr = jnp.asarray(R0)
    P_gas0 = P_AMB + 2.0 * SIGMA / R0
    return BubbleState(R=R0_arr, R0=R0_arr, P_gas0=jnp.asarray(P_gas0))


@pytest.fixture
def expanded_state():
    """BubbleState at R = 1.5 R0 with positive velocity."""
    R0_arr = jnp.asarray(R0)
    P_gas0 = P_AMB + 2.0 * SIGMA / R0
    return BubbleState(
        R=jnp.asarray(1.5 * R0),
        R_dot=jnp.asarray(0.5),
        R0=R0_arr,
        P_gas0=jnp.asarray(P_gas0),
    )


@pytest.fixture
def compressed_state():
    """BubbleState at R = 0.8 R0 with negative velocity."""
    R0_arr = jnp.asarray(R0)
    P_gas0 = P_AMB + 2.0 * SIGMA / R0
    return BubbleState(
        R=jnp.asarray(0.8 * R0),
        R_dot=jnp.asarray(-0.3),
        R0=R0_arr,
        P_gas0=jnp.asarray(P_gas0),
    )


@pytest.fixture
def simple_eom():
    """KellerMiksis with polytropic gas, no shell, Newtonian medium."""
    return KellerMiksis(
        gas=PolytropicGas(gamma=GAMMA),
        shell=NoShell(sigma=SIGMA),
        medium=NewtonianMedium(mu=MU),
        R0=R0,
        P_amb=P_AMB,
        rho_L=RHO_L,
        c_L=C_L,
    )


@pytest.fixture
def rp_eom():
    """RayleighPlesset with polytropic gas, no shell, Newtonian medium."""
    return RayleighPlesset(
        gas=PolytropicGas(gamma=GAMMA),
        shell=NoShell(sigma=SIGMA),
        medium=NewtonianMedium(mu=MU),
        R0=R0,
        P_amb=P_AMB,
        rho_L=RHO_L,
    )


@pytest.fixture
def sine_pulse():
    """5-cycle sine tone burst at 1 MHz, 100 kPa."""
    return ToneBurst(freq=FREQ, pressure=PRESSURE, shape=Sine(), cycle_num=5)
