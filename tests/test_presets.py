"""Tests for jbubble.utils.presets."""

import jax
import pytest
from jbubble import SaveSpec, run_simulation
from jbubble.bubble.eom import EquationOfMotion, KellerMiksis
from jbubble.bubble.shell import LipidShell, ThickShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.base import Pulse
from jbubble.utils.presets import (
    BubblePreset,
    free_bubble,
    lipid_bubble,
    thick_shell_bubble,
)


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


class TestFreeBubble:
    def test_default_params(self):
        eom, pulse = free_bubble()
        assert isinstance(eom, KellerMiksis)
        assert isinstance(pulse, ToneBurst)
        assert float(eom.R0) == pytest.approx(2e-6, rel=1e-10)

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
        eom, pulse = lipid_bubble()
        assert isinstance(eom.shell, LipidShell)

    def test_custom_params(self):
        eom, _ = lipid_bubble(R0=1.5e-6, kappa_s=3e-9, chi=0.6)
        assert float(eom.R0) == pytest.approx(1.5e-6, rel=1e-10)

    def test_simulation_runs(self):
        preset = lipid_bubble()
        result = jax.jit(run_simulation)(
            preset.eom,
            preset.pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )
        assert bool(result.converged)


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
