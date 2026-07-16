"""Tests for jbubble.bubble.gas."""

import jax
import jax.numpy as jnp
import pytest
from jbubble.bubble.gas import PolytropicGas, VanDerWaalsGas
from jbubble.bubble.state import BubbleState

P_GAS0 = 173_325.0  # typical equilibrium gas pressure
R0 = 2e-6


@pytest.fixture
def eq_state():
    return BubbleState(
        R=jnp.asarray(R0),
        R0=jnp.asarray(R0),
        P_gas0=jnp.asarray(P_GAS0),
    )


def _make_state(R, R_dot=0.0):
    return BubbleState(
        R=jnp.asarray(R),
        R_dot=jnp.asarray(R_dot),
        R0=jnp.asarray(R0),
        P_gas0=jnp.asarray(P_GAS0),
    )


class TestPolytropicGas:
    def test_equilibrium_returns_P_gas0(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        assert float(gas(eq_state)) == pytest.approx(P_GAS0, rel=1e-10)

    def test_expansion_lowers_pressure(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        expanded = _make_state(1.5 * R0)
        assert float(gas(expanded)) < P_GAS0

    def test_compression_raises_pressure(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        compressed = _make_state(0.8 * R0)
        assert float(gas(compressed)) > P_GAS0

    def test_isothermal_gamma_1(self, eq_state):
        gas = PolytropicGas(gamma=1.0)
        s = _make_state(1.5 * R0)
        expected = P_GAS0 * (R0 / (1.5 * R0)) ** 3
        assert float(gas(s)) == pytest.approx(expected, rel=1e-10)

    def test_adiabatic_gamma_1_4(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        R = 1.2 * R0
        s = _make_state(R)
        expected = P_GAS0 * (R0 / R) ** (3 * 1.4)
        assert float(gas(s)) == pytest.approx(expected, rel=1e-10)

    def test_jit_compatible(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        result = jax.jit(gas)(eq_state)
        assert float(result) == pytest.approx(P_GAS0, rel=1e-10)

    def test_grad_wrt_R(self, eq_state):
        gas = PolytropicGas(gamma=1.4)
        grad = jax.grad(gas)(eq_state)
        # dp/dR at equilibrium should be negative (expansion lowers pressure)
        assert float(grad.R) < 0


class TestVanDerWaalsGas:
    def test_equilibrium_returns_P_gas0(self, eq_state):
        gas = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        assert float(gas(eq_state)) == pytest.approx(P_GAS0, rel=1e-10)

    def test_zero_h_frac_recovers_polytropic(self):
        gas_vdw = VanDerWaalsGas(gamma=1.4, h_frac=0.0)
        gas_poly = PolytropicGas(gamma=1.4)
        s = _make_state(1.3 * R0)
        assert float(gas_vdw(s)) == pytest.approx(float(gas_poly(s)), rel=1e-8)

    def test_hard_core_raises_pressure_more(self):
        """VanDerWaals gives higher pressure than polytropic at same compression."""
        gas_vdw = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        gas_poly = PolytropicGas(gamma=1.4)
        s = _make_state(0.7 * R0)  # significant compression
        assert float(gas_vdw(s)) > float(gas_poly(s))

    def test_jit_compatible(self, eq_state):
        gas = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        result = jax.jit(gas)(eq_state)
        assert jnp.isfinite(result)

    def test_grad_wrt_R(self, eq_state):
        gas = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
        grad = jax.grad(gas)(eq_state)
        assert float(grad.R) < 0


# ── ThermalGas ──────────────────────────────────────────────────────────────
from jbubble.bubble.gas import ThermalGas  # noqa: E402
from jbubble.bubble.state import ThermalBubbleState  # noqa: E402

T0 = 293.0


def _thermal_state(R, R_dot=0.0, T=T0):
    return ThermalBubbleState(
        R=jnp.asarray(R),
        R_dot=jnp.asarray(R_dot),
        R0=jnp.asarray(R0),
        P_gas0=jnp.asarray(P_GAS0),
        T=jnp.asarray(T),
        T0=jnp.asarray(T0),
    )


class TestThermalGas:
    def test_equilibrium_returns_P_gas0(self):
        gas = ThermalGas(gamma=1.4, thermal_time=1e-7, T_amb=T0)
        s = _thermal_state(R0, T=T0)  # R=R0, T=T0
        assert float(gas(s)) == pytest.approx(P_GAS0, rel=1e-10)

    def test_isothermal_pressure(self):
        """At T = T0 the law is the isothermal polytropic p = P0 (R0/R)^3."""
        gas = ThermalGas(gamma=1.4, thermal_time=1e-7, T_amb=T0)
        R = 1.3 * R0
        s = _thermal_state(R, T=T0)
        assert float(gas(s)) == pytest.approx(P_GAS0 * (R0 / R) ** 3, rel=1e-10)

    def test_adiabatic_limit_matches_polytropic(self):
        """With T set to the adiabatic value, p matches P0 (R0/R)^(3 gamma)."""
        gamma = 1.4
        gas = ThermalGas(gamma=gamma, thermal_time=1e9, T_amb=T0)
        R = 1.3 * R0
        T_adiab = T0 * (R0 / R) ** (3 * (gamma - 1))
        s = _thermal_state(R, T=T_adiab)
        assert float(gas(s)) == pytest.approx(
            P_GAS0 * (R0 / R) ** (3 * gamma), rel=1e-8
        )

    def test_compression_heats_gas(self):
        """Adiabatic term: compressing (R_dot < 0) drives dT/dt > 0."""
        gas = ThermalGas(gamma=1.4, thermal_time=1e9, T_amb=T0)  # tau huge -> no relax
        s = _thermal_state(R0, R_dot=-1.0, T=T0)
        assert float(gas._dT_dt(s)) > 0.0

    def test_relaxation_restores_T0(self):
        """With R_dot = 0, a hot gas cools back toward T0 (dT/dt < 0)."""
        gas = ThermalGas(gamma=1.4, thermal_time=1e-7, T_amb=T0)
        s = _thermal_state(R0, R_dot=0.0, T=1.2 * T0)
        assert float(gas._dT_dt(s)) < 0.0

    def test_d_state_only_sets_T(self):
        gas = ThermalGas(gamma=1.4, thermal_time=1e-7, T_amb=T0)
        s = _thermal_state(1.1 * R0, R_dot=0.5, T=1.1 * T0)
        d = gas.d_state(s)
        assert isinstance(d, ThermalBubbleState)
        assert float(d.R) == 0.0 and float(d.R_dot) == 0.0
        assert float(d.R0) == 0.0 and float(d.P_gas0) == 0.0 and float(d.T0) == 0.0
        assert float(d.T) == pytest.approx(float(gas._dT_dt(s)))

    def test_augment_initial_state(self):
        from jbubble.bubble.state import BubbleState

        gas = ThermalGas(gamma=1.4, thermal_time=1e-7, T_amb=T0)
        base = BubbleState(R=jnp.asarray(R0), R0=jnp.asarray(R0), P_gas0=jnp.asarray(P_GAS0))
        s = gas.augment_initial_state(base)
        assert isinstance(s, ThermalBubbleState)
        assert float(s.T) == pytest.approx(T0) and float(s.T0) == pytest.approx(T0)

    def test_jit_and_grad(self):
        gas = ThermalGas(gamma=1.4, thermal_time=1e-7, T_amb=T0)
        s = _thermal_state(1.2 * R0, T=1.05 * T0)
        assert jnp.isfinite(jax.jit(gas)(s))
        g = jax.grad(gas)(s)
        assert float(g.R) < 0.0  # expansion lowers pressure
        assert float(g.T) > 0.0  # higher T raises pressure


class TestThermalDampingIntegration:
    """End-to-end: the resolved temperature dissipates energy (thermal damping).

    Integrates a µm-scale free decay through Keller-Miksis, which needs float64
    (rtol ≈ 1e-8) to stay stable.  The pytest session runs float32 (a plugin
    initialises jax before jbubble enables x64), so these are skipped there and
    run in a proper f64 environment.
    """

    def _free_decay_amplitude_ratio(self, gas):
        import diffrax
        import equinox as eqx
        from jbubble import SaveSpec, SolverConfig, run_simulation
        from jbubble.bubble.eom import KellerMiksis
        from jbubble.bubble.medium import NewtonianMedium
        from jbubble.bubble.shell import NoShell
        from jbubble.pulse.sampled import SampledPulse

        r0 = 2e-6
        eom = KellerMiksis(
            gas=gas, shell=NoShell(sigma=0.0), medium=NewtonianMedium(mu=1e-4),
            R0=r0, P_amb=101_325.0, rho_L=998.0, c_L=1500.0,
        )
        # start displaced 10% from equilibrium, then let it ring down freely
        s0 = eqx.tree_at(lambda s: s.R, eom.initial_state(), jnp.asarray(1.1 * r0))
        tmax = 6e-6
        ts = jnp.linspace(0.0, tmax, 50)
        pulse = SampledPulse(ts=ts, pressures=jnp.zeros_like(ts))
        cfg = SolverConfig(
            solver=diffrax.Dopri5(),
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-11),
            max_steps=1_000_000,
        )
        R = jnp.asarray(
            run_simulation(
                eom, pulse, config=cfg, t_max=tmax, state0=s0,
                save_spec=SaveSpec(num_samples=1000),
            ).radius
        )
        # ratio of late to early oscillation amplitude — smaller = more damped
        early = jnp.max(jnp.abs(R[:170] - r0))
        late = jnp.max(jnp.abs(R[-170:] - r0))
        return float(late / early)

    def test_thermal_gas_damps_ring_down(self):
        """A resolved temperature at ωτ~1 dissipates energy that the reversible
        polytropic gas cannot: the free ring-down decays far faster."""
        if not jax.config.jax_enable_x64:
            pytest.skip("µm-scale KM free decay needs float64")
        thermal = self._free_decay_amplitude_ratio(
            ThermalGas(gamma=1.4, thermal_time=1e-7)
        )
        polytropic = self._free_decay_amplitude_ratio(PolytropicGas(gamma=1.4))
        assert thermal < 0.5 * polytropic  # strong, robust separation
