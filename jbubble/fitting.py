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

PyTree = Any  # any JAX-compatible pytree (scalar, array, dict, eqx.Module, …)


def _format_params(params: PyTree) -> str:
    """Format a params pytree for logging — scalars shown as values, arrays as shapes."""
    parts = []
    for path, x in jax.tree_util.tree_leaves_with_path(params):
        name = jax.tree_util.keystr(path)
        val = f"{float(x):.4g}" if x.ndim == 0 else f"array{x.shape}"
        parts.append(f"{name}={val}" if name else val)
    return parts[0] if len(parts) == 1 else "[" + ", ".join(parts) + "]"


@dataclasses.dataclass
class FitResult:
    """Output of :func:`fit_parameters`.

    Attributes
    ----------
    params : PyTree
        Fitted parameter values — same type and structure as ``params0``.
    loss_history : jax.Array, shape (n_steps,)
        Loss value recorded at each optimisation step.
    result : SimulationResult
        Final forward simulation run with the fitted parameters.
    """

    params: PyTree
    loss_history: jax.Array
    result: SimulationResult


def fit_parameters(
    make_model: Callable[[PyTree], tuple[EquationOfMotion, Pulse]],
    params0: PyTree,
    *,
    save_spec: SaveSpec,
    t_max: float | None = None,
    loss_fn: Callable[[SimulationResult], jax.Array],
    optimizer: optax.GradientTransformation,
    n_steps: int = 200,
    config: SolverConfig | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    step_callback: Callable[[int, PyTree, float], None] | None = None,
    log_every: int = 25,
) -> FitResult:
    """Fit model parameters by differentiating through the ODE integration.

    ``make_model`` maps the current ``params`` to an ``(eom, pulse)`` pair,
    so anything that affects either the bubble physics or the acoustic drive
    can be optimised jointly from a single params pytree.  Examples::

        # Single parameter as a dict
        fit_parameters(
            make_model=lambda p: (make_eom(p), my_pulse),
            params0={'kappa_s': 2.4e-9},
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
    params0 : PyTree
        Initial parameter values — any JAX-compatible pytree (scalar,
        array, dict, tuple, or ``eqx.Module``).
    save_spec : SaveSpec
        Output sampling specification.
    t_max : float, optional
        Integration end time [s].  ``None`` uses ``pulse.t_end``.
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
        ``(step: int, params: PyTree, loss: float) → None``.  Called after
        each optimisation step in the Python loop (outside JIT), so Python
        side-effects like appending to a list work correctly.  Useful for
        recording parameter trajectories.
    log_every : int
        Print loss and current parameters every this many steps.
        Set to 0 to disable logging.  Default: 25.

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
        )
    if adjoint is None:
        adjoint = diffrax.RecursiveCheckpointAdjoint()

    # Partition params into array leaves (to be differentiated and tracked by
    # the optimiser) and static leaves (integer shapes, activation types, etc.)
    # that must not be traced.  This handles both plain jnp.array scalars and
    # eqx.Module params (e.g. a NeuralProperty network) transparently.
    array_params, static = eqx.partition(params0, eqx.is_array)

    def _loss(array_p: PyTree) -> jax.Array:
        params = eqx.combine(array_p, static)
        eom, pulse = make_model(params)
        sol = solve_eom(
            eom,
            pulse,
            t_max=t_max,
            save_spec=save_spec,
            config=config,
            adjoint=adjoint,
        )

        converged = diffrax.is_successful(sol.result)

        def _check_converged(c: bool) -> None:
            if not c:
                raise RuntimeError(
                    "ODE solver did not converge during fitting — the bubble may "
                    "have entered inertial cavitation (extreme collapse). "
                    "Consider reducing driving pressure."
                )

        jax.debug.callback(_check_converged, converged)

        # Solution converged -- assert for type safety
        assert sol.ts is not None
        assert sol.ys is not None

        ys = sol.ys
        ys_dot = jax.vmap(lambda t, x: eom(t, x, pulse))(sol.ts, ys)

        result = SimulationResult(
            ts=sol.ts,
            state=ys,
            state_dot=ys_dot,
            driving_pressure=jax.vmap(pulse)(sol.ts),
            converged=converged,
        )
        return loss_fn(result)

    loss_and_grad = eqx.filter_jit(jax.value_and_grad(_loss))

    opt_state = optimizer.init(array_params)
    loss_history = []

    for step in range(n_steps):
        loss_val, grads = loss_and_grad(array_params)
        loss_float = float(loss_val)

        updates, opt_state = optimizer.update(grads, opt_state)
        array_params = optax.apply_updates(array_params, updates)
        loss_history.append(loss_float)

        if step_callback is not None:
            step_callback(step, eqx.combine(array_params, static), loss_float)

        if log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            print(
                f"  step {step:>4} / {n_steps}  loss = {loss_float:.4e}"
                f"  params = {_format_params(array_params)}"
            )

    params = eqx.combine(array_params, static)
    eom, pulse = make_model(params)

    final_result = run_simulation(
        eom,
        pulse,
        save_spec=save_spec,
        t_max=t_max,
        config=config,
    )

    return FitResult(
        params=params,
        loss_history=jnp.array(loss_history),
        result=final_result,
    )
