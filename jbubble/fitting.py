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
    make_eom: Callable[..., EquationOfMotion],
    pulse: Pulse,
    params0: Any,
    *,
    save_spec: SaveSpec,
    t_span: tuple[float, float] | None = None,
    loss_fn: Callable[..., jax.Array],  # (state,) -> scalar
    optimizer: optax.GradientTransformation,
    n_steps: int = 200,
    config: SolverConfig | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    verbose: bool = True,
) -> FitResult:
    """Fit model parameters to a target radius-time curve.

    Minimises ``loss_fn(simulated_radius, target)`` with respect to
    ``params0`` by differentiating through the ODE integration.

    Uses an explicit solver (``Tsit5``) and ``BacksolveAdjoint`` by default.
    Explicit solvers avoid the ill-conditioned linear-system adjoint that
    implicit solvers (e.g. ``Kvaerno5``) can produce during backward passes.

    Parameters
    ----------
    make_eom : callable
        Factory that builds an ``EquationOfMotion`` from the current
        parameter values.  Must be JAX-traceable (use ``jnp`` operations
        on the parameters, not bare Python conditionals on JAX values).
    pulse : Pulse
        Driving acoustic pulse.
    params0 : Any
        Initial parameter values — any JAX-compatible pytree (scalar,
        array, tuple, or ``eqx.Module``).
    save_spec : SaveSpec
        Output sampling specification.  The simulated state is evaluated
        on this grid and passed to ``loss_fn``.
    t_span : tuple[float, float], optional
        Integration interval ``(t0, t1)`` [s].  ``None`` uses the pulse
        duration as reported by ``pulse.t_end``.
    loss_fn : callable
        ``(state: BubbleState) → scalar``.  Receives the full simulated
        state; close over any target data and reference constants.
        See :mod:`jbubble.metrics` for differentiable array utilities, e.g.::

            from jbubble.metrics import normalised_mse_radius
            loss_fn=lambda state: normalised_mse_radius(state.R, target, R0)
    optimizer : optax.GradientTransformation
        Gradient-based optimiser, e.g. ``optax.adam(1e-2)``.
    n_steps : int
        Number of optimisation steps.  Default: 200.
    config : SolverConfig, optional
        ODE solver settings.  Default: Tsit5 with PID(rtol=1e-4, atol=1e-8),
        50 000 max steps.
    adjoint : diffrax.AbstractAdjoint, optional
        Adjoint method.  Default: ``BacksolveAdjoint()`` (memory-efficient,
        works with explicit solvers).
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
            solver=diffrax.Tsit5(),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-8),
            dt0=1e-9,
            max_steps=50_000,
        )
    if adjoint is None:
        adjoint = diffrax.BacksolveAdjoint()

    # Partition params into array leaves (to be differentiated and tracked by
    # the optimiser) and static leaves (integer shapes, activation types, etc.)
    # that must not be traced.  This handles both plain jnp.array scalars and
    # eqx.Module params (e.g. a NeuralProperty network) transparently.
    array_params, static = eqx.partition(params0, eqx.is_array)

    def _loss(array_p: Any) -> jax.Array:
        params = eqx.combine(array_p, static)
        eom = make_eom(params)
        sol = solve_eom(
            eom,
            pulse,
            t_span=t_span,
            save_spec=save_spec,
            config=config,
            adjoint=adjoint,
        )
        return loss_fn(sol.ys)

    loss_and_grad = eqx.filter_jit(jax.value_and_grad(_loss))

    opt_state = optimizer.init(array_params)
    loss_history = []

    for step in range(n_steps):
        loss_val, grads = loss_and_grad(array_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        array_params = optax.apply_updates(array_params, updates)
        loss_history.append(float(loss_val))

        if verbose and (step % 25 == 0 or step == n_steps - 1):
            print(f"  step {step:>4} / {n_steps}  loss = {float(loss_val):.4e}")

    params = eqx.combine(array_params, static)

    final_result = run_simulation(
        make_eom(params),
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
