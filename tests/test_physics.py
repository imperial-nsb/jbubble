"""Physics regression tests against independent references.

The other test modules check each component against its own formula. These
tests check assembled equations of motion against results derived
independently of jbubble's code:

- the linearised resonance and damping of each equation of motion, against
  the natural frequency with surface tension (Brennen 1995, Eq. 4.8) and
  hand-derived linearisations of each equation;
- a driven Keller-Miksis trajectory, against a hand-coded right-hand side
  with explicit derivatives, integrated by SciPy.

References
----------
Brennen, C. E. (1995). *Cavitation and Bubble Dynamics*. Oxford University
Press, Section 4.2, Eq. 4.8.

Keller, J. B., & Miksis, M. (1980). Bubble oscillations of large amplitude.
*The Journal of the Acoustical Society of America*, 68(2), 628-633.
<https://doi.org/10.1121/1.384720>
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jbubble.bubble.eom import (
    KellerMiksis,
    ModifiedRayleighPlesset,
    RayleighPlesset,
)
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.bubble.state import BubbleState
from jbubble.pulse import ToneBurst
from jbubble.pulse.envelope import HannEnvelope
from jbubble.pulse.shapes import Sine
from jbubble.solver import SaveSpec, SolverConfig, solve_eom
from scipy.integrate import solve_ivp

R0 = 2e-6
P_AMB = 101325.0
RHO = 998.0
C = 1500.0
SIGMA = 0.072
KAPPA = 1.4

_SILENT = ToneBurst(freq=1e6, pressure=0.0, shape=Sine())


def _components(mu):
    return {
        "gas": PolytropicGas(gamma=KAPPA),
        "shell": NoShell(sigma=SIGMA),
        "medium": NewtonianMedium(mu=mu),
        "R0": R0,
        "P_amb": P_AMB,
        "rho_L": RHO,
    }


def _natural_frequency_squared():
    r"""Brennen (1995) Eq. 4.8 with no vapour pressure,
    $\omega_N^2 = [3\kappa P_\infty + 2(3\kappa - 1)\sigma/R_0] / (\rho R_0^2)$."""
    return (3 * KAPPA * P_AMB + 2 * (3 * KAPPA - 1) * SIGMA / R0) / (RHO * R0**2)


def _linearisation(eom):
    """Return (omega^2, 2 beta) from the Jacobian of the right-hand side at
    equilibrium, written as x'' + 2 beta x' + omega^2 x = 0."""
    y0 = eom.initial_state()

    def rhs(x):
        state = BubbleState(R=x[0], R_dot=x[1], R0=y0.R0, P_gas0=y0.P_gas0)
        d = eom(jnp.asarray(0.0), state, _SILENT)
        return jnp.stack([d.R, d.R_dot])

    J = np.asarray(jax.jacfwd(rhs)(jnp.stack([y0.R, y0.R_dot])))
    assert J[0, 0] == 0.0 and J[0, 1] == 1.0
    return -J[1, 0], -J[1, 1]


class TestLinearisedResonanceAndDamping:
    r"""Linearise each EoM about $R = R_0 + x$, $\dot{R} = \dot{x}$.

    With the polytropic gas, constant surface tension, and a Newtonian
    liquid, $p_L = P_{g0}(R_0/R)^{3\kappa} - 2\sigma/R - 4\mu\dot{R}/R$, so
    at equilibrium $\partial p_L/\partial R = -\rho R_0 \omega_N^2$ and
    $\partial p_L/\partial \dot{R} = -4\mu/R_0$, where $\omega_N$ is the
    natural frequency of Brennen's Eq. 4.8.
    """

    @pytest.mark.parametrize("mu", [0.0, 1e-3, 0.05])
    def test_rayleigh_plesset(self, mu):
        r"""$R_0\ddot{x} = (\partial_R p_L\, x + \partial_{\dot R} p_L\,\dot{x})/\rho$."""
        omega_sq, two_beta = _linearisation(RayleighPlesset(**_components(mu)))
        assert omega_sq == pytest.approx(_natural_frequency_squared(), rel=1e-12)
        assert two_beta == pytest.approx(4 * mu / (RHO * R0**2), rel=1e-12, abs=1e-6)

    @pytest.mark.parametrize("mu", [0.0, 1e-3])
    def test_modified_rayleigh_plesset_gas_radiation_damping(self, mu):
        r"""The $(R/c)\,\mathrm{d}p_\text{gas}/\mathrm{d}t$ term adds
        $3\kappa P_{g0}/(\rho c R_0)$ to $2\beta$ and leaves $\omega$ alone."""
        eom = ModifiedRayleighPlesset(**_components(mu), c_L=C)
        omega_sq, two_beta = _linearisation(eom)
        P_g0 = P_AMB + 2 * SIGMA / R0
        assert omega_sq == pytest.approx(_natural_frequency_squared(), rel=1e-12)
        expected = 4 * mu / (RHO * R0**2) + 3 * KAPPA * P_g0 / (RHO * C * R0)
        assert two_beta == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("mu", [0.0, 1e-3, 0.05])
    def test_keller_miksis_radiation_damping(self, mu):
        r"""Keller-Miksis to first order in $x$:

        $$
        R_0\left(1 + \frac{4\mu}{\rho c R_0}\right)\ddot{x}
            = -R_0\omega_N^2 x
            - \left(\frac{4\mu}{\rho R_0} + \frac{R_0^2\omega_N^2}{c}\right)\dot{x},
        $$

        so the inviscid radiation damping is $2\beta = \omega_N^2 R_0 / c$,
        that is, a damping rate $\omega_N^2 R_0 / (2c)$.
        """
        omega_sq, two_beta = _linearisation(KellerMiksis(**_components(mu), c_L=C))
        omega_N_sq = _natural_frequency_squared()
        stretch = 1 + 4 * mu / (RHO * C * R0)
        assert omega_sq == pytest.approx(omega_N_sq / stretch, rel=1e-12)
        expected = (4 * mu / (RHO * R0**2) + omega_N_sq * R0 / C) / stretch
        assert two_beta == pytest.approx(expected, rel=1e-12)


# ── hand-coded Keller-Miksis trajectory ──────────────────────────────────────


def _hann_tone_burst(P, f, cycles):
    """Return p(t) and dp/dt of a Hann-windowed sine burst in NumPy."""
    T = cycles / f

    def p(t):
        return P * np.sin(2 * np.pi * f * t) * 0.5 * (1 - np.cos(2 * np.pi * t / T))

    def dp(t):
        s, c = np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)
        w = 0.5 * (1 - np.cos(2 * np.pi * t / T))
        dw = 0.5 * (2 * np.pi / T) * np.sin(2 * np.pi * t / T)
        return P * (2 * np.pi * f * c * w + s * dw)

    return p, dp


def _keller_miksis_reference(P, mu, f, cycles, ts):
    r"""Integrate Keller & Miksis (1980) with hand-coded derivatives.

    With $p_L = P_{g0}(R_0/R)^{3\kappa} - 2\sigma/R - 4\mu\dot{R}/R$ and
    $M = \dot{R}/c$, solving

    $$
    (1 - M) R\ddot{R} + \tfrac{3}{2}(1 - M/3)\dot{R}^2
        = (1 + M)\frac{p_L - P_\text{amb} - p_\text{ac}}{\rho}
        + \frac{R}{\rho c}\left(\frac{\mathrm{d}p_L}{\mathrm{d}t}
        - \frac{\mathrm{d}p_\text{ac}}{\mathrm{d}t}\right)
    $$

    for $\ddot{R}$, where
    $\mathrm{d}p_L/\mathrm{d}t = \partial_R p_L\,\dot{R} + \partial_{\dot R} p_L\,\ddot{R}$.
    """
    p_ac, dp_ac = _hann_tone_burst(P, f, cycles)
    P_g0 = P_AMB + 2 * SIGMA / R0

    def rhs(t, y):
        R, V = y
        with np.errstate(invalid="ignore"):  # rejected trial steps can reach R < 0
            p_gas = P_g0 * (R0 / R) ** (3 * KAPPA)
        p_L = p_gas - 2 * SIGMA / R - 4 * mu * V / R
        dpL_dR = -3 * KAPPA * p_gas / R + 2 * SIGMA / R**2 + 4 * mu * V / R**2
        dpL_dV = -4 * mu / R
        M = V / C
        lhs = (1 - M) * R - R / (RHO * C) * dpL_dV
        rhs_ = (
            (1 + M) * (p_L - P_AMB - p_ac(t)) / RHO
            + R / (RHO * C) * (dpL_dR * V - dp_ac(t))
            - 1.5 * (1 - M / 3) * V**2
        )
        return [V, rhs_ / lhs]

    sol = solve_ivp(
        rhs,
        (0.0, ts[-1]),
        [R0, 0.0],
        method="DOP853",
        t_eval=ts,
        rtol=1e-12,
        atol=[1e-15 * R0, 1e-12],
    )
    assert sol.success
    return sol.y[0]


@pytest.mark.parametrize(("pressure", "mu"), [(100e3, 1e-3), (200e3, 1e-3)])
def test_keller_miksis_trajectory_matches_hand_coded_reference(pressure, mu):
    f, cycles = 1e6, 5
    pulse = ToneBurst(
        freq=f,
        pressure=pressure,
        shape=Sine(),
        cycle_num=cycles,
        envelope=HannEnvelope(),
    )
    eom = KellerMiksis(**_components(mu), c_L=C)
    config = SolverConfig(
        stepsize_controller=diffrax.PIDController(rtol=1e-11, atol=1e-13)
    )
    sol = solve_eom(
        eom, pulse, t_max=cycles / f, save_spec=SaveSpec(400), config=config
    )
    reference = _keller_miksis_reference(pressure, mu, f, cycles, np.asarray(sol.ts))
    R = np.asarray(sol.ys.R)
    assert R.max() / R0 > 1.2  # a nonlinear oscillation, not a linear one
    assert np.max(np.abs(R - reference)) / R0 < 1e-8
