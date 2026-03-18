"""Gradient-based parameter estimation for bubble dynamics models."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .bubble.eom import EquationOfMotion
from .pulse import Pulse
from .simulation import SimulationResult, run_simulation
from .solver import SaveSpec, SolverConfig, solve_eom


@dataclasses.dataclass
class FitResult:
    """Output of :func:`fit_parameters`.

    Attributes
    ----------
    params : Any
        Fitted parameter values — same type and structure as ``params0``.
    loss_history : jax.Array, shape (n_steps,)
        Loss value recorded at each optimisation step.
    result : SimulationResult
        Final forward simulation run with the fitted parameters.
    """

    params: Any
    loss_history: jax.Array
    result: SimulationResult


def fit_parameters(
    make_model: Callable[..., tuple[EquationOfMotion, Pulse]],
    params0: Any,
    *,
    save_spec: SaveSpec,
    t_span: tuple[float, float] | None = None,
    loss_fn: Callable[[SimulationResult], jax.Array],
    optimizer: optax.GradientTransformation,
    n_steps: int = 200,
    config: SolverConfig | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    step_callback: Callable[[int, Any, float], None] | None = None,
    verbose: bool = True,
) -> FitResult:
    """Fit model parameters by differentiating through the ODE integration.

    ``make_model`` maps the current ``params`` to an ``(eom, pulse)`` pair,
    so anything that affects either the bubble physics or the acoustic drive
    can be optimised jointly from a single params pytree.  Examples::

        # Fixed pulse — close over it
        fit_parameters(
            make_model=lambda kappa_s: (make_eom(kappa_s), my_pulse),
            params0=2.4e-9,
            ...
        )

        # Joint frequency + radius optimisation
        fit_parameters(
            make_model=lambda p: (make_eom(p['R0']), make_pulse(p['freq'])),
            params0={'R0': 5e-6, 'freq': 1e6},
            ...
        )

        # Heterogeneous — neural surface tension + scalar kappa_s + neural pulse
        fit_parameters(
            make_model=lambda p: (
                KellerMiksis(shell=LipidShell(sigma=p.sigma, kappa_s=p.kappa_s), ...),
                p.pulse,
            ),
            params0=LearnedParams(sigma=NeuralProperty(...), kappa_s=2.4e-9, pulse=NeuralPulse(...)),
            ...
        )

    Parameters
    ----------
    make_model : callable
        ``params → (EquationOfMotion, Pulse)``.  Must be JAX-traceable.
    params0 : Any
        Initial parameter values — any JAX-compatible pytree (scalar,
        array, dict, tuple, or ``eqx.Module``).
    save_spec : SaveSpec
        Output sampling specification.
    t_span : tuple[float, float], optional
        Integration interval ``(t0, t1)`` [s].  ``None`` uses the pulse
        duration as reported by ``pulse.t_end``.
    loss_fn : callable
        ``(result: SimulationResult) → scalar``.  Receives the full
        :class:`~jbubble.simulation.SimulationResult`; close over any
        target data and reference constants.
    optimizer : optax.GradientTransformation
        Gradient-based optimiser, e.g. ``optax.adam(1e-2)``.
    n_steps : int
        Number of optimisation steps.  Default: 200.
    config : SolverConfig, optional
        ODE solver settings.  Default: Tsit5 with PID(rtol=1e-4, atol=1e-8),
        50 000 max steps.
    adjoint : diffrax.AbstractAdjoint, optional
        Adjoint method.  Default: ``RecursiveCheckpointAdjoint()``
        (checkpoints the forward pass for stable gradients).
    step_callback : callable, optional
        ``(step: int, params: Any, loss: float) → None``.  Called after
        each optimisation step in the Python loop (outside JIT), so Python
        side-effects like appending to a list work correctly.  Useful for
        recording parameter trajectories.
    verbose : bool
        Print loss every 25 steps.  Default: True.

    Returns
    -------
    FitResult
        Fitted parameters, full loss history, and a final
        :class:`~jbubble.simulation.SimulationResult`.
    """
    if config is None:
        config = SolverConfig(
            solver=diffrax.Dopri5(),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-8),
            dt0=1e-9,
            max_steps=50_000,
        )
    if adjoint is None:
        adjoint = diffrax.RecursiveCheckpointAdjoint()

    # Partition params into array leaves (to be differentiated and tracked by
    # the optimiser) and static leaves (integer shapes, activation types, etc.)
    # that must not be traced.  This handles both plain jnp.array scalars and
    # eqx.Module params (e.g. a NeuralProperty network) transparently.
    array_params, static = eqx.partition(params0, eqx.is_array)

    def _loss(array_p: Any) -> jax.Array:
        params = eqx.combine(array_p, static)
        eom, pulse = make_model(params)
        sol = solve_eom(
            eom,
            pulse,
            t_span=t_span,
            save_spec=save_spec,
            config=config,
            adjoint=adjoint,
        )
        ys = sol.ys
        ys_dot = jax.vmap(lambda t, s: eom(t, s, pulse))(sol.ts, ys)
        result = SimulationResult(
            ts=sol.ts,
            state=ys,
            state_dot=ys_dot,
            driving_pressure=jax.vmap(pulse)(sol.ts),
            converged=diffrax.is_successful(sol.result),
        )
        return loss_fn(result)

    loss_and_grad = eqx.filter_jit(jax.value_and_grad(_loss))

    opt_state = optimizer.init(array_params)
    loss_history = []

    for step in range(n_steps):
        loss_val, grads = loss_and_grad(array_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        array_params = optax.apply_updates(array_params, updates)
        loss_history.append(float(loss_val))

        if step_callback is not None:
            step_callback(step, eqx.combine(array_params, static), float(loss_val))

        if verbose and (step % 25 == 0 or step == n_steps - 1):
            print(f"  step {step:>4} / {n_steps}  loss = {float(loss_val):.4e}")

    params = eqx.combine(array_params, static)
    eom, pulse = make_model(params)

    final_result = run_simulation(
        eom,
        pulse,
        save_spec=save_spec,
        t_span=t_span,
        config=config,
    )

    return FitResult(
        params=params,
        loss_history=jnp.array(loss_history),
        result=final_result,
    )
