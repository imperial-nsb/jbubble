"""Diffrax-based ODE solvers for bubble dynamics."""

from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

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

    def build(self, t0: jax.Array, t1: jax.Array) -> diffrax.SaveAt:
        ts = jnp.linspace(t0, t1, self.num_samples)
        return diffrax.SaveAt(ts=ts)


class SolverConfig(eqx.Module):
    """Numerical integration settings for :func:`solve_eom`.

    Fields
    ------
    solver : diffrax.AbstractSolver
        ODE solver.  Default: ``Dopri()``.
    stepsize_controller : diffrax.AbstractStepSizeController
        Step-size controller.  Default: ``PIDController(rtol=1e-5, atol=1e-6)``.
    dt0 : float
        Initial step size [s].  Default: 1e-9.
    max_steps : int
        Maximum solver steps per integration.  Default: 50 000.
    """

    solver: diffrax.AbstractSolver = eqx.field(default_factory=diffrax.Dopri5)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        default_factory=lambda: diffrax.PIDController(rtol=1e-6, atol=1e-9)
    )
    dt0: float = 1e-9
    max_steps: int = eqx.field(default=10_000, static=True)


def solve_eom(
    eom: EquationOfMotion,
    pulse: Pulse,
    *,
    y0: Any = None,
    t_max: ArrayLike | None = None,
    save_spec: SaveSpec | None = None,
    config: SolverConfig | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    progress: bool = False,
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
    t_max : float, optional
        Integration end time [s].  If *None*, derived from ``pulse.t_end``.
    save_spec : SaveSpec, optional
        Output sampling specification.  Default: 1024 evenly-spaced
        time points.
    config : SolverConfig, optional
        Numerical integration settings.  Default: Kvaerno5 with
        PIDController(rtol=1e-3, atol=1e-6).
    adjoint : diffrax.AbstractAdjoint, optional
        Adjoint method for gradient computation.  Default: diffrax built-in
        (``RecursiveCheckpointAdjoint``).  For gradient-based fitting through
        an explicit solver use ``diffrax.BacksolveAdjoint()``.
    progress : bool
        Show a text progress meter.

    Returns
    -------
    diffrax.Solution
        Solution object with ``ts`` and ``ys``.
    """
    if save_spec is None:
        save_spec = SaveSpec(num_samples=1024)
    assert isinstance(save_spec, SaveSpec)

    if config is None:
        config = SolverConfig()
    assert isinstance(config, SolverConfig)

    if y0 is None:
        y0 = eom.initial_state()

    t0 = jnp.asarray(0.0)
    t1 = jnp.asarray(pulse.t_end if t_max is None else t_max)
    saveat = save_spec.build(t0, t1)

    def ode_func(t, state, args):
        eom_model, pulse_model = args
        return eom_model(t, state, pulse_model)

    term = diffrax.ODETerm(ode_func)
    progress_meter = (
        diffrax.TextProgressMeter() if progress else diffrax.NoProgressMeter()
    )
    _adjoint = adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint()

    return diffrax.diffeqsolve(
        term,
        config.solver,
        t0=t0,
        t1=t1,
        dt0=config.dt0,
        y0=y0,
        args=(eom, pulse),
        saveat=saveat,
        stepsize_controller=config.stepsize_controller,
        max_steps=config.max_steps,
        progress_meter=progress_meter,
        throw=False,
        adjoint=_adjoint,
    )
