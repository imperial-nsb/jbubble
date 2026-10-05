"""Tests for jbubble.utils.presets."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jbubble import SaveSpec, run_simulation
from jbubble.bubble.eom import EquationOfMotion, KellerMiksis
from jbubble.bubble.shell import (
    LipidShell,
    MarmottantSurfaceTension,
    NoShell,
    SmoothMarmottantSurfaceTension,
    ThickShell,
)
from jbubble.bubble.state import BubbleState
from jbubble.pulse import ToneBurst
from jbubble.pulse.base import Pulse
from jbubble.utils.presets import (
    BubblePreset,
    free_bubble,
    lipid_bubble,
    thick_shell_bubble,
)


def _val(prop):
    """Return the value of a ConstantProperty as a float."""
    return float(prop.val)


def _linear_response(eom):
    """Return the linear resonance frequency [Hz] and quality factor."""
    y0 = eom.initial_state()

    def rhs(v):
        st = BubbleState(R=v[0], R_dot=v[1], R0=y0.R0, P_gas0=y0.P_gas0)
        d = eom(jnp.asarray(0.0), st, lambda t: 0.0 * t)
        return jnp.stack([d.R, d.R_dot])

    J = jax.jacfwd(rhs)(jnp.stack([y0.R, y0.R_dot]))
    w2, beta = -float(J[1, 0]), -float(J[1, 1]) / 2.0
    return math.sqrt(w2) / (2.0 * math.pi), math.sqrt(w2) / (2.0 * beta)


def _peak_expansion(eom, pulse):
    radius = run_simulation(eom, pulse).radius
    return float(jnp.max(radius)) / float(eom.R0) - 1.0


class TestBubblePreset:
    def test_is_named_tuple(self):
        preset = free_bubble()
        assert isinstance(preset, tuple)
        assert isinstance(preset, BubblePreset)

    def test_can_unpack(self):
        eom, pulse = free_bubble()
        assert isinstance(eom, EquationOfMotion)
        assert isinstance(pulse, Pulse)

    def test_named_access(self):
        preset = free_bubble()
        assert isinstance(preset.eom, EquationOfMotion)
        assert isinstance(preset.pulse, Pulse)

    @pytest.mark.parametrize("factory", [free_bubble, lipid_bubble, thick_shell_bubble])
    def test_presets_share_the_liquid(self, factory):
        eom, _ = factory()
        assert _val(eom.medium.mu) == 1e-3
        assert float(eom.P_amb) == 101325.0
        assert float(eom.rho_L) == 998.0
        assert float(eom.c_L) == 1500.0

    @pytest.mark.parametrize("factory", [free_bubble, lipid_bubble, thick_shell_bubble])
    def test_presets_start_at_equilibrium(self, factory):
        eom, _ = factory()
        y0 = eom.initial_state()
        d = eom(jnp.asarray(0.0), y0, lambda t: 0.0 * t)
        assert float(d.R_dot) == 0.0


class TestFreeBubble:
    def test_default_params(self):
        eom, pulse = free_bubble()
        assert isinstance(eom, KellerMiksis)
        assert isinstance(pulse, ToneBurst)
        assert isinstance(eom.shell, NoShell)
        assert float(eom.R0) == pytest.approx(2e-6, rel=1e-10)
        assert _val(eom.gas.gamma) == 1.4
        assert _val(eom.shell.sigma) == 0.072

    def test_custom_params(self):
        eom, pulse = free_bubble(R0=3e-6, freq=2e6, pressure=200e3)
        assert float(eom.R0) == pytest.approx(3e-6, rel=1e-10)

    def test_simulation_runs(self):
        preset = free_bubble()
        result = jax.jit(run_simulation)(
            preset.eom,
            preset.pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )
        assert bool(result.converged)
        assert result.ts.shape == (200,)


class TestLipidBubble:
    def test_default_params(self):
        # Gümmer et al. (2021): SonoVue with SF6, chi = 0.5 N/m,
        # kappa_s = 7.5e-9 kg/s, sigma_0 = 0.020 N/m, sigma_c = 0.072 N/m.
        eom, pulse = lipid_bubble()
        assert isinstance(eom, KellerMiksis)
        assert isinstance(eom.shell, LipidShell)
        sigma = eom.shell.sigma
        assert isinstance(sigma, SmoothMarmottantSurfaceTension)
        assert _val(sigma.chi) == 0.5
        assert _val(sigma.sigma_rupture) == 0.072
        assert float(sigma.smoothing) == 0.01
        assert _val(eom.shell.kappa_s) == 7.5e-9
        assert _val(eom.gas.gamma) == 1.095

    def test_default_buckling_ratio_gives_gummer_sigma_0(self):
        # Gümmer et al. (2021), Eq. 11: R_buckle_ratio = (1 + sigma_0 / chi)^(-1/2).
        assert round(1.0 / math.sqrt(1.0 + 0.020 / 0.5), 5) == 0.98058
        eom, _ = lipid_bubble()
        sigma_0 = float(eom.shell.sigma(eom.initial_state()))
        assert sigma_0 == pytest.approx(0.020, rel=1e-4)
        P_gas0 = float(eom.initial_state().P_gas0)
        assert P_gas0 == pytest.approx(101325.0 + 2.0 * sigma_0 / 2e-6, rel=1e-12)

    def test_custom_params(self):
        eom, _ = lipid_bubble(R0=1.5e-6, kappa_s=3e-9, chi=0.6, smoothing=0.005)
        assert float(eom.R0) == pytest.approx(1.5e-6, rel=1e-10)
        assert _val(eom.shell.sigma.chi) == 0.6
        assert float(eom.shell.sigma.smoothing) == 0.005

    def test_simulation_runs(self):
        preset = lipid_bubble()
        result = jax.jit(run_simulation)(
            preset.eom,
            preset.pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )
        assert bool(result.converged)

    def test_peak_expansion_matches_piecewise_marmottant(self):
        eom, pulse = lipid_bubble(pressure=50e3)
        smooth = eom.shell.sigma
        piecewise = MarmottantSurfaceTension(
            R_buckle_ratio=smooth.R_buckle_ratio,
            chi=smooth.chi,
            sigma_rupture=smooth.sigma_rupture,
        )
        eom_piecewise = eqx.tree_at(lambda m: m.shell.sigma, eom, piecewise)
        ref = _peak_expansion(eom_piecewise, pulse)
        assert _peak_expansion(eom, pulse) == pytest.approx(ref, rel=0.01)

    def test_gradient_with_respect_to_chi_is_finite(self):
        _, pulse = lipid_bubble(pressure=50e3)

        def peak(chi):
            eom, _ = lipid_bubble(pressure=50e3, chi=chi)
            radius = run_simulation(
                eom, pulse, save_spec=SaveSpec(num_samples=256)
            ).radius
            return jnp.max(radius)

        grad = jax.grad(peak)(jnp.asarray(0.5))
        assert bool(jnp.isfinite(grad))
        # A stiffer shell expands less.
        assert float(grad) < 0.0


class TestThickShellBubble:
    def test_default_params(self):
        eom, pulse = thick_shell_bubble()
        assert isinstance(eom.shell, ThickShell)

    def test_custom_params(self):
        eom, _ = thick_shell_bubble(R0=3e-6, d_s=20e-9, G_s=15e6)
        assert float(eom.R0) == pytest.approx(3e-6, rel=1e-10)

    def test_simulation_runs(self):
        preset = thick_shell_bubble()
        result = jax.jit(run_simulation)(
            preset.eom,
            preset.pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )
        assert bool(result.converged)
