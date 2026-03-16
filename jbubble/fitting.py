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
from .solver import SaveSpec, solve_eom


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
    target: jax.Array,
    params0: Any,
    *,
    save_spec: SaveSpec,
    window_s: float,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    optimizer: optax.GradientTransformation | None = None,
    n_steps: int = 200,
    solver: diffrax.AbstractSolver | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    dt0: float = 1e-9,
    max_steps: int = 50_000,
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
    target : jax.Array, shape (N,)
        Target radius-time curve sampled on the same grid as ``save_spec``.
    params0 : Any
        Initial parameter values — any JAX-compatible pytree (scalar,
        array, tuple, or ``eqx.Module``).
    save_spec : SaveSpec
        Output sampling specification.  The simulated radius is compared
        to ``target`` on this grid.
    window_s : float
        Simulation time window [s].
    loss_fn : callable, optional
        ``(r_sim, r_target) → scalar``.  Default: mean squared error.
        See :mod:`jbubble.metrics` for ready-made differentiable losses.
    optimizer : optax.GradientTransformation, optional
        Gradient-based optimiser.  Default: ``optax.adam(0.01)``.
    n_steps : int
        Number of optimisation steps.  Default: 200.
    solver : diffrax.AbstractSolver, optional
        ODE solver.  Default: ``Tsit5()`` (explicit; good adjoint conditioning).
    adjoint : diffrax.AbstractAdjoint, optional
        Adjoint method.  Default: ``BacksolveAdjoint()`` (memory-efficient,
        works with explicit solvers).
    dt0 : float
        Initial ODE step size [s].  Default: 1e-9.
    max_steps : int
        Maximum ODE steps per integration.  Default: 50 000.
    verbose : bool
        Print loss every 25 steps.  Default: True.

    Returns
    -------
    FitResult
        Fitted parameters, full loss history, and a final
        :class:`~jbubble.simulation.SimulationResult`.
    """
    if solver is None:
        solver = diffrax.Tsit5()
    if adjoint is None:
        adjoint = diffrax.BacksolveAdjoint()
    if optimizer is None:
        optimizer = optax.adam(0.01)
    if loss_fn is None:
        loss_fn = lambda r_sim, r_target: jnp.mean((r_sim - r_target) ** 2)  # noqa: E731

    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-8)

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
            t_span=(0.0, window_s),
            dt0=dt0,
            save_spec=save_spec,
            solver=solver,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            max_steps=max_steps,
        )
        return loss_fn(sol.ys.R, target)

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
        window_s=window_s,
        dt0=dt0,
        solver=solver,
    )

    return FitResult(
        params=params,
        loss_history=jnp.array(loss_history),
        result=final_result,
    )
