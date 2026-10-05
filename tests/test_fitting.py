"""Tests for jbubble.fitting.

Most tests fit a small Rayleigh-Plesset model over 3 us so that the fast
suite stays fast. Tests marked `slow` recover parameters to a tolerance.
"""

import warnings
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from jbubble import SaveSpec, run_simulation
from jbubble.bubble.eom import RayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.property import NeuralProperty
from jbubble.bubble.shell import NoShell
from jbubble.fitting import (
    FitResult,
    Parameter,
    _held_floats,
    _stack,
    fit_parameters,
    unwrap,
)
from jbubble.metrics import normalised_mse_radius
from jbubble.pulse import NeuralPulse, ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.simulation import SimulationResult
from jbubble.solver import SolverConfig

R0 = 2e-6
SAVE = SaveSpec(64)
T_MAX = 3e-6


def _pulse(pressure=30e3, cycles=3):
    return ToneBurst(freq=1e6, pressure=pressure, shape=Sine(), cycle_num=cycles)


def _eom(mu=1e-3, sigma=0.072, R0_=R0):
    return RayleighPlesset(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=sigma),
        medium=NewtonianMedium(mu=mu),
        R0=R0_,
        P_amb=101325.0,
        rho_L=998.0,
    )


def _radius(mu=1e-3, pressure=30e3):
    return run_simulation(
        _eom(mu), _pulse(pressure), save_spec=SAVE, t_max=T_MAX
    ).radius


@pytest.fixture(scope="module")
def target():
    return _radius()


class _Learned(eqx.Module):
    mu: Any
    sigma: Any = 0.072


def _learned_model(p):
    return _eom(p.mu, sigma=p.sigma), _pulse()


def _fit(target, **kwargs):
    options = dict(
        make_model=lambda p: (_eom(p["mu"]), _pulse()),
        params0={"mu": 2e-3},
        loss_fn=lambda r: normalised_mse_radius(r.radius, target, R0),
        optimizer=optax.adam(0.05),
        n_steps=3,
        save_spec=SAVE,
        t_max=T_MAX,
        log_every=0,
    )
    options.update(kwargs)
    return fit_parameters(**options)


# ── Parameter ────────────────────────────────────────────────────────────────


class TestParameter:
    @pytest.mark.parametrize(
        "parameter, value",
        [
            (Parameter(2.4e-9), 2.4e-9),
            (Parameter(-3.0), -3.0),
            (Parameter(2.4e-9, lower=0.0), 2.4e-9),
            (Parameter(-0.2, upper=0.0), -0.2),
            (Parameter(0.5, lower=0.0, upper=1.7), 0.5),
            (Parameter(0.0, scale=1e-7), 0.0),
            (
                Parameter(jnp.array([1e-6, 2e-6, 4e-6]), lower=0.0),
                jnp.array([1e-6, 2e-6, 4e-6]),
            ),
            (Parameter(2e-6, fixed=True), 2e-6),
            (Parameter(0.0, fixed=True), 0.0),  # no scale needed
            (Parameter(jnp.zeros(3), fixed=True), jnp.zeros(3)),
        ],
    )
    def test_value_round_trip(self, parameter, value):
        assert jnp.allclose(parameter.value, jnp.asarray(value), rtol=1e-12, atol=0)
        assert parameter.value.dtype == jnp.float64

    @pytest.mark.parametrize(
        "parameter",
        [
            Parameter(2.4e-9),
            Parameter(2.4e-9, lower=0.0),
            Parameter(2e-6, lower=1e-6, upper=4e-6),
        ],
    )
    def test_coordinate_is_order_one(self, parameter):
        assert abs(float(parameter.raw)) <= 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(value=2.0, lower=0.0, upper=1.0),
            dict(value=-1.0, lower=0.0),
            dict(value=1.0, upper=0.0),
            dict(value=0.5, lower=1.0, upper=1.0),
            dict(value=0.0),
            dict(value=1.0, scale=-1.0),
            dict(value=float("nan")),
        ],
            dict(value=0.5, lower=float("nan")),
            dict(value=0.5, upper=float("nan")),
            dict(value=0.5, lower=float("inf")),
            dict(value=0.5, upper=float("-inf")),
    )
    def test_invalid_specification_raises(self, kwargs):
        with pytest.raises(ValueError):
            Parameter(**kwargs)

    def test_bounds_hold_for_any_coordinate(self):
    @pytest.mark.parametrize(
        "kwargs, lower, upper",
        [
            (dict(lower=0.0, upper=float("inf")), 0.0, None),
            (dict(lower=-jnp.inf, upper=1.0), None, 1.0),
            (dict(lower=-jnp.inf, upper=jnp.inf), None, None),
        ],
    )
    def test_infinite_bound_means_no_bound(self, kwargs, lower, upper):
        p = Parameter(0.5, **kwargs)
        assert (p.lower, p.upper) == (lower, upper)
        assert bool(jnp.isfinite(p.raw))
        assert float(p.value) == pytest.approx(0.5, rel=1e-12)

        u = jnp.linspace(-50.0, 50.0, 101)
        bounded = Parameter(0.5, lower=0.1, upper=1.7)
        positive = Parameter(1e-9, lower=0.0)
        values = jax.vmap(lambda r: eqx.tree_at(lambda p: p.raw, bounded, r).value)(u)
        assert bool(jnp.all((values >= 0.1) & (values <= 1.7)))
        values = jax.vmap(lambda r: eqx.tree_at(lambda p: p.raw, positive, r).value)(u)
        assert bool(jnp.all(values >= 0.0))

    def test_relative_scale_makes_gradients_unit_free(self):
        # d f / d u = x * d f / d x for a lower bound at zero, whatever the units.
        g_small = jax.grad(lambda p: p.value * 1e9)(Parameter(1e-9, lower=0.0)).raw
        g_large = jax.grad(lambda p: p.value)(Parameter(1.0, lower=0.0)).raw
        assert jnp.allclose(g_small, g_large)

    def test_fixed_parameter_has_no_gradient(self):
        g = jax.grad(lambda p: 3.0 * p.value)(Parameter(2.0, fixed=True)).raw
        assert float(g) == 0.0

    def test_unwrap_dict_and_module(self):
        tree = {"a": Parameter(2.0, lower=0.0), "shell": NoShell(sigma=Parameter(0.07))}
        out = unwrap(tree)
        assert jnp.allclose(out["a"], 2.0)
        assert jnp.allclose(out["shell"].sigma.val, 0.07)


# ── what gets fitted ─────────────────────────────────────────────────────────


class TestWhatGetsFitted:
    def test_python_float_leaves_are_fitted(self, target):
        fit = _fit(target)
        assert isinstance(fit.params["mu"], jax.Array)
        assert float(fit.params["mu"]) != 2e-3

    def test_root_float_is_fitted(self, target):
        fit = _fit(target, make_model=lambda mu: (_eom(mu), _pulse()), params0=2e-3)
        assert float(fit.params) != 2e-3

    def test_non_float_leaves_are_held_fixed(self, target):
        fit = _fit(
            target,
            params0={
                "mu": 2e-3,
                "n": 3,
                "flag": True,
                "name": "x",
                "sigma": Parameter(0.07, fixed=True),
            },
            make_model=lambda p: (_eom(p["mu"], sigma=p["sigma"]), _pulse()),
        )
        assert (
            fit.params["n"] == 3
            and fit.params["flag"] is True
            and fit.params["name"] == "x"
        )
        assert float(fit.params["sigma"]) == 0.07

    def test_float_fields_inside_modules_are_static(self):
        net = eqx.nn.MLP(1, 1, 4, 1, key=jax.random.PRNGKey(0))
        pulse0 = NeuralPulse(net=net, pulse_duration=3e-6, pressure_scale=30e3)
        with warnings.catch_warnings():
            # initial_time and the envelope's steepness keep their defaults,
            # so the held-float warning stays quiet.
            warnings.simplefilter("error", UserWarning)
            fit = fit_parameters(
                lambda p: (_eom(), p),
                pulse0,
                loss_fn=lambda r: jnp.mean((r.radius / R0 - 1.05) ** 2),
                optimizer=optax.adam(1e-2),
                n_steps=2,
                save_spec=SaveSpec(32),
                t_max=T_MAX,
                log_every=0,
            )
        assert fit.params.pulse_duration == 3e-6 and fit.params.pressure_scale == 30e3
        assert not eqx.tree_equal(fit.params.net, net)

    def test_set_float_inside_module_warns_and_is_held(self, target):
        params0 = _Learned(mu=Parameter(2e-3, lower=0.0), sigma=0.07)
        with pytest.warns(
            UserWarning, match=r"holding fixed 1 Python float.*: sigma\."
        ):
            fit = _fit(target, params0=params0, make_model=_learned_model)
        assert fit.params.sigma == 0.07
        assert float(fit.params.mu) != 2e-3

    @pytest.mark.parametrize(
        "sigma",
        [None, Parameter(0.07, fixed=True)],
        ids=["default", "fixed-parameter"],
    )
    def test_default_or_fixed_float_inside_module_is_quiet(self, target, sigma):
        mu = Parameter(2e-3, lower=0.0)
        params0 = _Learned(mu=mu) if sigma is None else _Learned(mu=mu, sigma=sigma)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fit = _fit(target, params0=params0, make_model=_learned_model, n_steps=1)
        expected = 0.072 if sigma is None else 0.07
        assert float(fit.params.sigma) == pytest.approx(expected, rel=1e-12)

    def test_float_in_container_field_of_module_is_named(self):
        params0 = _Learned(mu=Parameter(2e-3, lower=0.0), sigma={"k": 2.5})
        assert _held_floats(params0) == ["sigma.k"]

    def test_weakly_typed_array_compiles_once(self, target):
        traces = []

        def make_model(mu):
            traces.append(mu)
            return _eom(mu), _pulse()

        _fit(
            target,
            make_model=make_model,
            params0=jnp.asarray(2e-3),  # weakly typed
            optimizer=optax.adam(1e-4),
        )
        assert len(traces) == 1

    def test_nothing_to_fit_raises_value_error(self, target):
        with pytest.raises(ValueError, match="nothing to fit"):
            _fit(target, params0={"n": 3, "mu": Parameter(1e-3, fixed=True)})

    def test_zero_float_names_the_entry(self, target):
        with pytest.raises(ValueError, match="'delay'"):
            _fit(target, params0={"mu": 2e-3, "delay": 0.0})


# ── API errors ───────────────────────────────────────────────────────────────


class TestErrors:
    def test_make_model_must_return_a_pair(self, target):
        with pytest.raises(TypeError, match="make_model must return"):
            _fit(target, make_model=lambda p: _eom(p["mu"]))

    def test_loss_must_be_scalar(self, target):
        with pytest.raises(TypeError, match="scalar"):
            _fit(target, loss_fn=lambda r: (r.radius - target) / R0)

    def test_make_model_without_condition_argument(self, target):
        with pytest.raises(TypeError, match=r"make_model\(params, condition\)"):
            _fit(target, conditions=[{"pressure": 30e3}])

    def test_condition_arguments_without_conditions(self, target):
        with pytest.raises(TypeError, match="pass conditions"):
            _fit(target, make_model=lambda p, c: (_eom(p["mu"]), _pulse()))

    def test_conditions_must_be_a_sequence(self, target):
        with pytest.raises(TypeError, match="list or tuple"):
            _fit(target, conditions={"pressure": jnp.array([20e3, 30e3])})


# ── result and callback ──────────────────────────────────────────────────────


class TestResult:
    def test_loss_history_matches_final_result(self, target):
        fit = _fit(target, n_steps=4)
        assert isinstance(fit, FitResult)
        assert fit.loss_history.shape == (5,)
        final = normalised_mse_radius(fit.result.radius, target, R0)
        assert jnp.allclose(fit.loss_history[-1], final, rtol=1e-12)
        assert bool(fit.result.converged)
        assert fit.num_rejected == 0 and not fit.stopped_early
        assert fit.message == "completed 4 steps"

    def test_step_callback_sees_params0_and_each_step(self, target):
        seen = []
        fit = _fit(
            target,
            step_callback=lambda s, p, loss: seen.append((s, float(p["mu"]), loss)),
        )
        assert [s for s, _, _ in seen] == [0, 1, 2, 3]
        assert seen[0][1] == pytest.approx(2e-3, rel=1e-12)
        assert all(
            loss == pytest.approx(float(fit.loss_history[s])) for s, _, loss in seen
        )

    def test_step_callback_stop_iteration_ends_fit(self, target):
        def callback(step, params, loss):
            if step == 2:
                raise StopIteration

        fit = _fit(target, n_steps=10, step_callback=callback)
        assert fit.stopped_early and fit.loss_history.shape == (3,)
        assert "StopIteration" in fit.message

    @pytest.mark.parametrize(
        ("params0", "lr", "message"),
        [(jnp.asarray(2e-3), 1e-1, "larger than"), (2e-3, 1e-12, "barely changes")],
    )
    def test_first_step_scale_warning(self, target, params0, lr, message):
        with pytest.warns(UserWarning, match=message):
            _fit(
                target,
                make_model=lambda mu: (_eom(mu), _pulse()),
                params0=params0,
                optimizer=optax.adam(lr),
                n_steps=1,
                max_backtracks=0,
            )

    def test_no_warning_for_well_scaled_parameters(self, target):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _fit(
                target,
                params0={
                    "mu": Parameter(2e-3, lower=0.0),
                    "s": Parameter(0.07, lower=0.0, upper=0.2),
                },
                make_model=lambda p: (_eom(p["mu"], sigma=p["s"]), _pulse()),
                n_steps=1,
            )


# ── failures ─────────────────────────────────────────────────────────────────


def _nan_above(threshold):
    """A loss that is NaN once the fitted R0 exceeds ``threshold``."""

    def loss_fn(result, target):
        base = normalised_mse_radius(result.radius, target, R0)
        return base + jnp.where(result.state.R0[0] > threshold, jnp.nan, 0.0)

    return loss_fn


class TestFailures:
    def test_non_finite_candidate_is_halved_not_applied(self, target):
        loss_fn = _nan_above(2.05e-6)
        with warnings.catch_warnings():
            # The optimum lies beyond the NaN wall, so the fit may stop early.
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = _fit(
                target,
                make_model=lambda p: (_eom(R0_=p["R0_um"] * 1e-6), _pulse()),
                params0={"R0_um": jnp.asarray(2.0)},
                loss_fn=lambda r: loss_fn(r, target * 1.2),
                optimizer=optax.sgd(5.0),
                n_steps=4,
            )
        assert fit.num_rejected > 0
        assert bool(jnp.all(jnp.isfinite(fit.loss_history)))
        assert float(fit.params["R0_um"]) <= 2.05

    def test_all_halvings_fail_warns_and_returns_last_good(self, target):
        loss_fn = _nan_above(2.0e-6 + 1e-15)  # any increase of R0 is rejected
        with pytest.warns(RuntimeWarning, match="stopped after 0 of 3 steps"):
            fit = _fit(
                target,
                make_model=lambda p: (_eom(R0_=p["R0_um"] * 1e-6), _pulse()),
                params0={"R0_um": jnp.asarray(2.0)},
                loss_fn=lambda r: loss_fn(r, target * 1.2),
                optimizer=optax.sgd(5.0),
                n_steps=3,
                max_backtracks=2,
            )
        assert fit.stopped_early and fit.num_rejected == 3
        assert float(fit.params["R0_um"]) == 2.0
        assert bool(fit.result.converged)

    def test_failure_at_params0_raises_runtime_error(self, target):
        with pytest.raises(RuntimeError, match="did not converge"):
            _fit(target, config=SolverConfig(max_steps=5))

    def test_non_finite_loss_at_params0_points_at_loss_fn(self, target):
        with pytest.raises(RuntimeError, match="the loss is nan.*check that loss_fn"):
            _fit(target, loss_fn=lambda r: jnp.log(-jnp.mean(r.radius)))

    def test_failure_at_params0_names_the_condition(self, target):
        with pytest.raises(
            RuntimeError, match="condition 1: the ODE solve did not converge"
        ):
            _fit(
                target,
                make_model=lambda p, c: (
                    _eom(p["mu"]),
                    _pulse(c["pressure"], cycles=c["cycles"]),
                ),
                conditions=[
                    {"pressure": 30e3, "cycles": 3},
                    {"pressure": 30e3, "cycles": 20},
                ],
                loss_fn=lambda r, c: jnp.mean((r.radius / R0 - 1.0) ** 2),
                t_max=None,
                config=SolverConfig(max_steps=700),
            )

    @pytest.mark.filterwarnings("ignore:fit_parameters. the first step:UserWarning")
    def test_failed_solve_mid_fit_is_backtracked(self, target):
        # With t_max=None the solve runs to pulse.t_end, so the step count grows
        # with cycle_num: 3 cycles take about 350 steps and 8 take about 960.
        # The loss -t_end pushes cycle_num up by 2 per cycle, so SGD(10)
        # proposes 3 -> 23 cycles.  With max_steps=700, the candidates at 23, 13,
        # and 8 cycles fail, and 5.5 cycles is accepted.
        fit = _fit(
            target,
            make_model=lambda p: (_eom(), _pulse(cycles=p["cycles"])),
            params0={"cycles": jnp.asarray(3.0)},
            loss_fn=lambda r: -r.ts[-1] * 1e6,
            optimizer=optax.sgd(10.0),
            n_steps=1,
            t_max=None,
            config=SolverConfig(max_steps=700),
        )
        assert fit.num_rejected == 3 and not fit.stopped_early
        assert float(fit.params["cycles"]) == pytest.approx(5.5)
        assert bool(fit.result.converged)


# ── several conditions ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def two_pressures():
    return [{"pressure": p, "radius": _radius(pressure=p)} for p in (20e3, 30e3)]


class TestConditions:
    def test_result_is_one_simulation_per_condition(self, two_pressures):
        fit = fit_parameters(
            lambda p, c: (_eom(p["mu"]), _pulse(c["pressure"])),
            {"mu": 2e-3},
            conditions=two_pressures,
            loss_fn=lambda r, c: normalised_mse_radius(r.radius, c["radius"], R0),
            optimizer=optax.adam(0.05),
            n_steps=3,
            save_spec=SAVE,
            t_max=T_MAX,
            log_every=0,
        )
        assert isinstance(fit.result, list) and len(fit.result) == 2
        assert all(
            isinstance(r, SimulationResult) and r.radius.shape == (64,)
            for r in fit.result
        )
        assert fit.loss_history[-1] < fit.loss_history[0]
        mean = jnp.mean(
            jnp.stack(
                [
                    normalised_mse_radius(r.radius, c["radius"], R0)
                    for r, c in zip(fit.result, two_pressures, strict=True)
                ]
            )
        )
        assert jnp.allclose(fit.loss_history[-1], mean, rtol=1e-12)

    def test_batched_and_sequential_conditions_agree(self, two_pressures):
        def run(conditions):
            return fit_parameters(
                lambda p, c: (_eom(p["mu"]), _pulse(c["pressure"])),
                {"mu": 2e-3},
                conditions=conditions,
                loss_fn=lambda r, c: normalised_mse_radius(r.radius, c["radius"], R0),
                optimizer=optax.adam(0.05),
                n_steps=3,
                save_spec=SAVE,
                t_max=T_MAX,
                log_every=0,
            )

        batched = run(two_pressures)  # same structure and shapes: vmap
        sequential = run(
            [dict(c, name=f"c{i}") for i, c in enumerate(two_pressures)]
        )  # strings: loop
        assert jnp.allclose(
            batched.loss_history, sequential.loss_history, rtol=1e-9, atol=0
        )
        assert jnp.allclose(
            batched.params["mu"], sequential.params["mu"], rtol=1e-9, atol=0
        )

    def test_python_bool_keeps_conditions_sequential(self, two_pressures):
        conditions = [dict(c, viscous=i == 0) for i, c in enumerate(two_pressures)]
        assert _stack(two_pressures) is not None
        assert _stack(conditions) is None
        fit = fit_parameters(
            lambda p, c: (_eom(p["mu"] if c["viscous"] else 1e-3), _pulse()),
            {"mu": 2e-3},
            conditions=conditions,
            loss_fn=lambda r, c: normalised_mse_radius(r.radius, c["radius"], R0),
            optimizer=optax.adam(0.05),
            n_steps=1,
            save_spec=SAVE,
            t_max=T_MAX,
            log_every=0,
        )
        assert len(fit.result) == 2

    def test_conditions_with_different_lengths(self):
        frames = [jnp.linspace(0.5e-6, 2.5e-6, n) for n in (20, 35)]
        full = _radius()
        ts = jnp.linspace(0.0, T_MAX, 64)
        conditions = [{"t": t, "radius": jnp.interp(t, ts, full)} for t in frames]
        fit = fit_parameters(
            lambda p, c: (_eom(p["mu"]), _pulse()),
            {"mu": 2e-3},
            conditions=conditions,
            loss_fn=lambda r, c: normalised_mse_radius(
                jnp.interp(c["t"], r.ts, r.radius), c["radius"], R0
            ),
            optimizer=optax.adam(0.05),
            n_steps=3,
            save_spec=SAVE,
            t_max=T_MAX,
            log_every=0,
        )
        assert len(fit.result) == 2 and fit.loss_history[-1] < fit.loss_history[0]


# ── optax and neural components ──────────────────────────────────────────────


def test_optimizer_receives_params(target):
    # optax.keep_params_nonnegative needs params; adamw's weight decay must not
    # move a fixed Parameter.
    optimizer = optax.chain(
        optax.adamw(1e-4, weight_decay=1e-2), optax.keep_params_nonnegative()
    )
    fit = _fit(
        target,
        params0={"mu": jnp.asarray(2e-3), "sigma": Parameter(0.072, fixed=True)},
        make_model=lambda p: (_eom(p["mu"], sigma=p["sigma"]), _pulse()),
        optimizer=optimizer,
    )
    assert float(fit.params["mu"]) >= 0.0 and float(fit.params["sigma"]) == 0.072


def test_neural_property_weights_are_fitted(target):
    mlp = eqx.nn.MLP(
        1, 1, 4, 1, final_activation=jax.nn.softplus, key=jax.random.PRNGKey(0)
    )

    def make_model(sigma):
        eom = RayleighPlesset(
            gas=PolytropicGas(gamma=1.4),
            shell=NoShell(sigma=sigma),
            medium=NewtonianMedium(mu=1e-3),
            R0=R0,
            P_amb=101325.0,
            rho_L=998.0,
        )
        return eom, _pulse()

    fit = _fit(
        target,
        make_model=make_model,
        params0=NeuralProperty(net=mlp),
        optimizer=optax.adam(1e-2),
    )
    assert isinstance(fit.params, NeuralProperty)
    assert fit.loss_history[-1] < fit.loss_history[0]


@pytest.mark.slow
def test_recovers_shared_parameters_from_three_pressures():
    pressures = (20e3, 40e3, 60e3)
    conditions = [
        {"pressure": p, "radius": _radius(mu=1.5e-3, pressure=p)} for p in pressures
    ]
    fit = fit_parameters(
        lambda p, c: (_eom(p["mu"], sigma=p["sigma"]), _pulse(c["pressure"])),
        {
            "mu": Parameter(3e-3, lower=0.0),
            "sigma": Parameter(0.05, lower=0.0, upper=0.2),
        },
        conditions=conditions,
        loss_fn=lambda r, c: normalised_mse_radius(r.radius, c["radius"], R0),
        optimizer=optax.adam(optax.cosine_decay_schedule(0.1, 150)),
        n_steps=150,
        save_spec=SAVE,
        t_max=T_MAX,
        log_every=0,
    )
    assert float(fit.params["mu"]) == pytest.approx(1.5e-3, rel=1e-2)
    assert float(fit.params["sigma"]) == pytest.approx(0.072, rel=1e-2)
