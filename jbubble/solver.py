"""Diffrax-based ODE solvers for bubble dynamics."""

from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import jax.numpy as jnp

from .bubble.eom import EquationOfMotion
from .pulse import Pulse


class SaveSpec(eqx.Module):
    """Specification for ODE output sampling.

    Parameters
    ----------
    num_samples : int
        Number of evenly-spaced time points to record.  Default: 1024.
    """

    num_samples: int = eqx.field(default=1024, static=True)

    def build(self, t0: float, t1: float) -> diffrax.SaveAt:
        ts = jnp.linspace(t0, t1, self.num_samples)
        return diffrax.SaveAt(ts=ts)


def solve_eom(
    eom: EquationOfMotion,
    pulse: Pulse,
    *,
    y0: Any = None,
    t_span: tuple[float, float] | None = None,
    dt0: float = 1e-3,
    save_spec: SaveSpec | None = None,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    progress: bool = False,
    max_steps: int = 10_000,
) -> diffrax.Solution:
    """Solve bubble dynamics for an ``EquationOfMotion``.

    Parameters
    ----------
    eom : EquationOfMotion
        Assembled equation of motion (e.g. ``KellerMiksis``).
    pulse : Pulse
        Driving acoustic pulse.
    y0 : BubbleState, optional
        Initial state (dimensionless).  If *None*, derived from
        ``eom.initial_state()``.
    t_span : tuple[float, float], optional
        Integration interval ``(t0, t1)``.  If *None*, derived from
        the pulse duration.
    dt0 : float
        Initial time step.
    save_spec : SaveSpec, optional
        Output sampling specification.  Default: 1024 evenly-spaced
        time points.
    solver : diffrax.AbstractSolver, optional
        ODE solver.  Default: ``Kvaerno5()``.
    stepsize_controller : diffrax.AbstractStepSizeController, optional
        Step-size controller.  Default:
        ``PIDController(rtol=1e-3, atol=1e-6)``.
    adjoint : diffrax.AbstractAdjoint, optional
        Adjoint method for gradient computation.  Default: diffrax built-in
        (``RecursiveCheckpointAdjoint``).  For gradient-based fitting through
        an explicit solver use ``diffrax.BacksolveAdjoint()``.
    progress : bool
        Show a text progress meter.
    max_steps : int
        Maximum number of solver steps.

    Returns
    -------
    diffrax.Solution
        Solution object with ``ts`` and ``ys``.
    """
    if t_span is None:
        t_span = (0.0, pulse.t_end)

    if save_spec is None:
        save_spec = SaveSpec(num_samples=1024)

    if solver is None:
        solver = diffrax.Kvaerno5()

    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

    if y0 is None:
        y0 = eom.initial_state()

    t0, t1 = t_span
    saveat = save_spec.build(t0, t1)

    def ode_func(t, state, args):
        eom_model, pulse_model = args
        return eom_model(t, state, pulse_model)

    term = diffrax.ODETerm(ode_func)
    progress_meter = (
        diffrax.TextProgressMeter() if progress else diffrax.NoProgressMeter()
    )

    adjoint_kwargs = {} if adjoint is None else {"adjoint": adjoint}

    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=(eom, pulse),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        progress_meter=progress_meter,
        throw=False,
        **adjoint_kwargs,
    )
