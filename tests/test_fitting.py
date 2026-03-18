"""Tests for jbubble.fitting."""

import diffrax
import jax
import jax.numpy as jnp
import optax
import pytest

from jbubble import SaveSpec, run_simulation
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.fitting import FitResult, fit_parameters
from jbubble.metrics import normalised_mse_radius
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.solver import SolverConfig

# Use low pressure and the implicit solver to avoid convergence issues
_PRESSURE = 20e3
_SAVE_SPEC = SaveSpec(num_samples=100)
_T_MAX = 5e-6
_CONFIG = SolverConfig(max_steps=50_000)
_PULSE = ToneBurst(freq=1e6, pressure=_PRESSURE, shape=Sine(), cycle_num=3)


def _make_eom(mu):
    return KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=mu),
        R0=2e-6,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


@pytest.mark.slow
class TestFitParameters:
    """Integration tests for the fitting pipeline."""

    def _target_radius(self):
        eom = _make_eom(mu=1e-3)
        result = jax.jit(run_simulation)(
            eom, _PULSE, save_spec=_SAVE_SPEC, t_max=_T_MAX
        )
        return result.radius

    def test_basic_fit_reduces_loss(self):
        target_radius = self._target_radius()

        def make_model(mu):
            return _make_eom(mu), _PULSE

        fit_result = fit_parameters(
            make_model=make_model,
            params0=jnp.asarray(1.5e-3),  # close-ish initial guess
            save_spec=_SAVE_SPEC,
            t_max=_T_MAX,
            loss_fn=lambda result: normalised_mse_radius(
                result.radius, target_radius, 2e-6
            ),
            optimizer=optax.adam(1e-5),
            n_steps=10,
            config=_CONFIG,
            log_every=0,
        )

        assert isinstance(fit_result, FitResult)
        assert fit_result.loss_history.shape == (10,)
        # Loss should decrease
        assert float(fit_result.loss_history[-1]) < float(
            fit_result.loss_history[0]
        )

    def test_fit_result_has_simulation(self):
        target_radius = self._target_radius()

        def make_model(mu):
            return _make_eom(mu), _PULSE

        fit_result = fit_parameters(
            make_model=make_model,
            params0=jnp.asarray(1.5e-3),
            save_spec=_SAVE_SPEC,
            t_max=_T_MAX,
            loss_fn=lambda result: normalised_mse_radius(
                result.radius, target_radius, 2e-6
            ),
            optimizer=optax.adam(1e-5),
            n_steps=5,
            config=_CONFIG,
            log_every=0,
        )
        assert fit_result.result.ts.shape == (100,)
        assert bool(fit_result.result.converged)

    def test_step_callback_called(self):
        target_radius = self._target_radius()

        def make_model(mu):
            return _make_eom(mu), _PULSE

        steps_seen = []

        def callback(step, params, loss):
            steps_seen.append(step)

        fit_parameters(
            make_model=make_model,
            params0=jnp.asarray(1.5e-3),
            save_spec=_SAVE_SPEC,
            t_max=_T_MAX,
            loss_fn=lambda result: normalised_mse_radius(
                result.radius, target_radius, 2e-6
            ),
            optimizer=optax.adam(1e-5),
            n_steps=5,
            config=_CONFIG,
            log_every=0,
            step_callback=callback,
        )
        assert len(steps_seen) == 5
        assert steps_seen == [0, 1, 2, 3, 4]

    def test_dict_params(self):
        target_radius = self._target_radius()

        def make_model(params):
            eom = KellerMiksis(
                gas=PolytropicGas(gamma=1.4),
                shell=NoShell(sigma=params["sigma"]),
                medium=NewtonianMedium(mu=params["mu"]),
                R0=2e-6,
                P_amb=101325.0,
                rho_L=998.0,
                c_L=1500.0,
            )
            return eom, _PULSE

        fit_result = fit_parameters(
            make_model=make_model,
            params0={"sigma": jnp.asarray(0.05), "mu": jnp.asarray(1.5e-3)},
            save_spec=_SAVE_SPEC,
            t_max=_T_MAX,
            loss_fn=lambda result: normalised_mse_radius(
                result.radius, target_radius, 2e-6
            ),
            optimizer=optax.adam(1e-5),
            n_steps=5,
            config=_CONFIG,
            log_every=0,
        )
        assert isinstance(fit_result.params, dict)
        assert "sigma" in fit_result.params
        assert "mu" in fit_result.params
