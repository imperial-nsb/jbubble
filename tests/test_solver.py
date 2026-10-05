"""Tests for jbubble.solver."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import LipidShell, MarmottantSurfaceTension, NoShell
from jbubble.bubble.state import BubbleState
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.simulation import run_simulation
from jbubble.solver import SaveSpec, SolverConfig, solve_eom

_PID = diffrax.PIDController


def _scaled(state, scale):
    return jtu.tree_map(jnp.divide, state, scale)


def _marmottant(kappa_s):
    sigma = MarmottantSurfaceTension(R_buckle_ratio=0.98, chi=0.55, sigma_rupture=0.072)
    return LipidShell(sigma=sigma, kappa_s=kappa_s)


def _lipid_bubble(kappa_s, *, gamma=1.07, R0=2e-6):
    return KellerMiksis(
        gas=PolytropicGas(gamma=gamma),
        shell=_marmottant(kappa_s),
        medium=NewtonianMedium(mu=1e-3),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


def _free_bubble(mu, *, R0=2e-6):
    return KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=mu),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


def _tone(pressure=100e3, cycles=5, freq=1e6):
    return ToneBurst(freq=freq, pressure=pressure, shape=Sine(), cycle_num=cycles)


class TestSaveSpec:
    def test_default_num_samples(self):
        spec = SaveSpec()
        assert spec.num_samples == 1024

    def test_custom_num_samples(self):
        spec = SaveSpec(num_samples=500)
        assert spec.num_samples == 500

    def test_build(self):
        spec = SaveSpec(num_samples=100)
        saveat = spec.build(jnp.asarray(0.0), jnp.asarray(10e-6))
        assert isinstance(saveat, diffrax.SaveAt)


class TestSolverConfig:
    def test_defaults(self):
        config = SolverConfig()
        assert isinstance(config.solver, diffrax.Dopri5)
        assert isinstance(config.stepsize_controller, diffrax.PIDController)
        assert config.stepsize_controller.rtol == 1e-6
        assert config.stepsize_controller.atol == 1e-10
        assert config.dt0 == 1e-9
        assert config.max_steps == 10_000

    def test_custom_config(self):
        config = SolverConfig(
            solver=diffrax.Kvaerno5(),
            dt0=1e-10,
            max_steps=50_000,
        )
        assert isinstance(config.solver, diffrax.Kvaerno5)
        assert config.dt0 == 1e-10
        assert config.max_steps == 50_000


class TestSolveEom:
    def test_returns_solution(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=100),
            t_max=5e-6,
        )
        assert isinstance(sol, diffrax.Solution)

    def test_solution_has_correct_shape(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=200),
            t_max=5e-6,
        )
        assert sol.ts.shape == (200,)
        assert sol.ys.R.shape == (200,)

    def test_converges(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=100),
            t_max=5e-6,
        )
        assert diffrax.is_successful(sol.result)

    def test_returns_state_in_si_units(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom, sine_pulse, save_spec=SaveSpec(num_samples=100), t_max=5e-6
        )
        y0 = simple_eom.initial_state()
        assert float(sol.ys.R[0]) == float(y0.R)
        assert jnp.all(sol.ys.R0 == y0.R0)
        assert jnp.allclose(sol.ys.P_gas0, y0.P_gas0, rtol=1e-15, atol=0)

    def test_uses_pulse_t_end_when_no_t_max(self, simple_eom, sine_pulse):
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            save_spec=SaveSpec(num_samples=100),
        )
        expected_t_end = float(sine_pulse.t_end)
        assert float(sol.ts[-1]) == pytest.approx(expected_t_end, rel=1e-6)


class TestScaledState:
    """The solver integrates `state / eom.state_scale(state)`."""

    def test_state_scale(self, simple_eom):
        y0 = simple_eom.initial_state()
        scale = simple_eom.state_scale(y0)
        assert float(scale.R) == float(y0.R0)
        assert float(scale.R0) == float(y0.R0)
        assert float(scale.R_dot) == pytest.approx((101325.0 / 998.0) ** 0.5)
        assert float(scale.P_gas0) == float(y0.P_gas0)

    def test_state_scale_falls_back_to_P_amb(self, simple_eom):
        y0 = eqx.tree_at(lambda s: s.P_gas0, simple_eom.initial_state(), 0.0)
        assert float(simple_eom.state_scale(y0).P_gas0) == 101325.0

    def test_frozen_fields_round_trip_exactly(self, simple_eom):
        """`(x / x) * x == x`, so the solver sees `R0` and `P_gas0` exactly."""
        y0 = simple_eom.initial_state(R=1.3 * simple_eom.R0)
        scale = simple_eom.state_scale(y0)
        back = jtu.tree_map(jnp.multiply, _scaled(y0, scale), scale)
        assert float(back.R0) == float(y0.R0)
        assert float(back.P_gas0) == float(y0.P_gas0)

    def test_step_sequence_is_independent_of_bubble_size(self):
        """An inviscid, tension-free Keller-Miksis bubble is self-similar.

        Scaling R0, t, and 1/f by 1/32 (exact in binary floating point) maps
        the problem onto itself, so with tolerances on R/R0 the solver takes
        exactly the same steps for both sizes. With an absolute tolerance in
        metres, it would not.
        """

        def solve(scale):
            eom = KellerMiksis(
                gas=PolytropicGas(gamma=1.4),
                shell=NoShell(sigma=0.0),
                medium=NewtonianMedium(mu=0.0),
                R0=2e-6 * scale,
                P_amb=101325.0,
                rho_L=998.0,
                c_L=1500.0,
            )
            pulse = _tone(150e3, cycles=3, freq=1e6 / scale)
            config = SolverConfig(dt0=1e-9 * scale)
            return solve_eom(
                eom, pulse, t_max=4e-6 * scale, save_spec=SaveSpec(256), config=config
            )

        big, small = solve(1.0), solve(1.0 / 32.0)
        assert int(big.stats["num_steps"]) == int(small.stats["num_steps"])
        assert int(big.stats["num_rejected_steps"]) == int(
            small.stats["num_rejected_steps"]
        )
        assert jnp.allclose(big.ys.R / 2e-6, small.ys.R / (2e-6 / 32), rtol=1e-13)

    def test_matches_an_si_state_solve(self):
        """At tight tolerances the scaled solve matches a solve of the SI state."""
        eom, pulse = _lipid_bubble(5e-9), _tone(200e3)
        config = SolverConfig(stepsize_controller=_PID(rtol=1e-11, atol=1e-13))
        scaled = solve_eom(eom, pulse, save_spec=SaveSpec(400), config=config)
        y0 = eom.initial_state()
        t0, t1 = jnp.asarray(0.0), jnp.asarray(pulse.t_end)
        si = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: eom(t, y, pulse)),
            diffrax.Tsit5(),
            t0,
            t1,
            1e-9,
            y0,
            saveat=SaveSpec(400).build(t0, t1),
            stepsize_controller=_PID(rtol=1e-11, atol=1e-13 * 2e-6),
            max_steps=1_000_000,
        )
        assert float(jnp.max(jnp.abs(scaled.ys.R - si.ys.R))) < 5e-8 * 2e-6
        velocity_error = jnp.max(jnp.abs(scaled.ys.R_dot - si.ys.R_dot))
        assert float(velocity_error) < 1e-6 * float(jnp.max(jnp.abs(si.ys.R_dot)))

    @pytest.mark.slow
    def test_gradients_through_the_scale_match_finite_differences(self):
        """`R0`, `P_amb`, and `rho_L` also set the (stop-gradient) scale."""
        pulse = _tone(150e3)
        config = SolverConfig(stepsize_controller=_PID(rtol=1e-11, atol=1e-13))
        theta0 = np.array([2e-6, 101325.0, 998.0])

        def loss(theta):
            eom = KellerMiksis(
                gas=PolytropicGas(gamma=1.4),
                shell=NoShell(sigma=0.072),
                medium=NewtonianMedium(mu=1e-3),
                R0=theta[0],
                P_amb=theta[1],
                rho_L=theta[2],
                c_L=1500.0,
            )
            sol = solve_eom(eom, pulse, save_spec=SaveSpec(400), config=config)
            return jnp.mean((sol.ys.R / 2e-6) ** 2)

        grad = np.asarray(jax.jit(jax.grad(loss))(jnp.asarray(theta0)))
        f = jax.jit(loss)
        for i in range(3):
            h = theta0[i] * 1e-5
            up, down = theta0.copy(), theta0.copy()
            up[i] += h
            down[i] -= h
            fd = (float(f(jnp.asarray(up))) - float(f(jnp.asarray(down)))) / (2 * h)
            assert grad[i] == pytest.approx(fd, rel=1e-5)


class TestInitialStateFilling:
    def test_partial_state_gets_equilibrium_fields(self, simple_eom, sine_pulse):
        R0 = simple_eom.R0
        sol = solve_eom(
            simple_eom,
            sine_pulse,
            y0=BubbleState(R=jnp.asarray(1.2 * R0)),
            save_spec=SaveSpec(100),
            t_max=5e-6,
        )
        y_eq = simple_eom.initial_state()
        assert diffrax.is_successful(sol.result)
        assert float(sol.ys.R[0]) == pytest.approx(1.2 * R0, rel=1e-14)
        assert jnp.all(sol.ys.R0 == y_eq.R0)
        assert jnp.allclose(sol.ys.P_gas0, y_eq.P_gas0, rtol=1e-15, atol=0)

    def test_matches_initial_state_keywords(self, simple_eom, sine_pulse):
        R0 = simple_eom.R0

        def solve(y0):
            return solve_eom(
                simple_eom, sine_pulse, y0=y0, save_spec=SaveSpec(100), t_max=5e-6
            )

        partial = solve(BubbleState(R=jnp.asarray(1.2 * R0), R_dot=jnp.asarray(0.5)))
        full = solve(simple_eom.initial_state(R=1.2 * R0, R_dot=0.5))
        assert jnp.array_equal(partial.ys.R, full.ys.R)

    def test_explicit_fields_are_kept(self, simple_eom, sine_pulse):
        y0 = BubbleState(
            R=jnp.asarray(3e-6),
            R0=jnp.asarray(3e-6),
            P_gas0=jnp.asarray(2e5),
        )
        sol = solve_eom(
            simple_eom, sine_pulse, y0=y0, save_spec=SaveSpec(10), t_max=1e-7
        )
        assert jnp.all(sol.ys.R0 == 3e-6)
        assert jnp.allclose(sol.ys.P_gas0, 2e5, rtol=1e-15, atol=0)

    def test_filled_P_gas0_uses_the_given_R0(self, simple_eom, sine_pulse):
        """A set R0 with an unset P_gas0 gets the Laplace pressure at that R0."""
        y0 = BubbleState(R=jnp.asarray(3e-6), R0=jnp.asarray(3e-6))
        sol = solve_eom(
            simple_eom, sine_pulse, y0=y0, save_spec=SaveSpec(10), t_max=1e-7
        )
        expected = 101325.0 + 2.0 * 0.072 / 3e-6
        assert jnp.allclose(sol.ys.P_gas0, expected, rtol=1e-14, atol=0)

    def test_works_under_jit(self, simple_eom, sine_pulse):
        @jax.jit
        def final_radius(R):
            sol = solve_eom(
                simple_eom,
                sine_pulse,
                y0=BubbleState(R=R),
                save_spec=SaveSpec(10),
                t_max=1e-6,
            )
            return sol.ys.R0[-1], sol.ys.P_gas0[-1]

        R0, P_gas0 = final_radius(jnp.asarray(2.2e-6))
        assert float(R0) == simple_eom.R0
        assert float(P_gas0) == pytest.approx(
            float(simple_eom.initial_state().P_gas0), rel=1e-15
        )


class TestNoHostCallbacks:
    """jbubble adds no host callbacks to a traced solve.

    The jaxpr does contain equinox's `pure_callback` error branch from inside
    diffrax (`eqx.error_if`), which runs only if diffrax raises.
    """

    def test_solve_eom_jaxpr_has_no_debug_callback(self, simple_eom, sine_pulse):
        jaxpr = jax.make_jaxpr(
            lambda e, p: solve_eom(e, p, save_spec=SaveSpec(16), t_max=1e-6).ys.R
        )(simple_eom, sine_pulse)
        assert "debug_callback" not in str(jaxpr)

    def test_run_simulation_jaxpr_has_no_debug_callback(self, simple_eom, sine_pulse):
        jaxpr = jax.make_jaxpr(
            lambda e, p: run_simulation(e, p, save_spec=SaveSpec(16), t_max=1e-6).radius
        )(simple_eom, sine_pulse)
        assert "debug_callback" not in str(jaxpr)
