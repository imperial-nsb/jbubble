"""High-level helpers for running and post-processing bubble dynamics simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .bubble.base import BubbleState, ConfinedBubbleState
from .bubble.eom import EquationOfMotion
from .pulse import Pulse
from .solver import SaveSpec, solve_eom


class SimulationResult(eqx.Module):
    """Output of :func:`run_simulation`.

    All array quantities are in SI units and sampled at ``ts``.

    Attributes
    ----------
    ts : jax.Array, shape (N,)
        Time points [s].
    radius : jax.Array, shape (N,)
        Bubble wall radius R(t) [m].
    radial_velocity : jax.Array, shape (N,)
        Bubble wall velocity dR/dt [m/s].
    radial_acceleration : jax.Array, shape (N,)
        Bubble wall acceleration d2R/dt2 [m/s^2], computed analytically
        from the ODE right-hand side (not numerical differentiation).
    vessel_radius : jax.Array or None, shape (N,)
        Vessel wall radius [m].  Non-``None`` only for
        :class:`~jbubble.bubble.SphericalConfinement`.
    vessel_velocity : jax.Array or None, shape (N,)
        Vessel wall velocity [m/s].  Non-``None`` only for
        :class:`~jbubble.bubble.SphericalConfinement`.
    driving_pressure : jax.Array, shape (N,)
        Applied acoustic pressure at the bubble [Pa].
    converged : jax.Array
        Boolean scalar; ``True`` if the ODE solver converged successfully.
    eom : EquationOfMotion
        The equation of motion used.
    pulse : Pulse
        The driving pulse used.
    """

    ts: jax.Array
    radius: jax.Array
    radial_velocity: jax.Array
    radial_acceleration: jax.Array
    vessel_radius: jax.Array | None
    vessel_velocity: jax.Array | None
    driving_pressure: jax.Array
    converged: jax.Array
    eom: EquationOfMotion
    pulse: Pulse

    @property
    def has_vessel(self) -> bool:
        return self.vessel_radius is not None


def run_simulation(
    eom: EquationOfMotion,
    pulse: Pulse,
    *,
    save_spec: SaveSpec,
    state0: Any = None,
    window_s: float = 20e-6,
    dt0: float = 1e-9,
    max_steps: int = 10_000,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    progress: bool = False,
) -> SimulationResult:
    """Run a simulation and return results in SI units.

    Parameters
    ----------
    eom : EquationOfMotion
        Assembled equation of motion (e.g. ``KellerMiksis``).
    pulse : Pulse
        Driving pulse.
    save_spec : SaveSpec
        Output specification (number of samples).
    state0 : BubbleState, optional
        Initial state.  Defaults to ``eom.initial_state()``.
    window_s : float
        Simulation time window [s].
    dt0 : float
        Initial time step [s].
    max_steps : int
        Maximum ODE steps.
    stepsize_controller : diffrax.AbstractStepSizeController, optional
        Step-size controller.  Default: ``PIDController(rtol=1e-3, atol=1e-6)``.
    progress : bool
        Show progress meter.

    Returns
    -------
    SimulationResult
        Results with time, radius, velocity, pressure, convergence info.
    """
    if state0 is None:
        state0 = eom.initial_state()

    sol = solve_eom(
        eom,
        pulse,
        y0=state0,
        t_span=(0.0, window_s),
        dt0=dt0,
        save_spec=save_spec,
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        progress=progress,
        max_steps=max_steps,
    )

    assert sol.ts is not None
    assert sol.ys is not None
    ys = sol.ys
    driving_pressure = jax.vmap(pulse)(sol.ts)

    # Compute acceleration analytically from the ODE RHS.
    def _rddot(t, state):
        return eom(t, state, pulse).R_dot

    rddot = jax.vmap(_rddot)(sol.ts, ys)

    has_vessel = isinstance(ys, ConfinedBubbleState)

    return SimulationResult(
        ts=sol.ts,
        radius=ys.R,
        radial_velocity=ys.R_dot,
        radial_acceleration=rddot,
        vessel_radius=ys.a if has_vessel else None,
        vessel_velocity=ys.a_dot if has_vessel else None,
        driving_pressure=driving_pressure,
        converged=diffrax.is_successful(sol.result),
        eom=eom,
        pulse=pulse,
    )


def compute_radius_metrics(result: SimulationResult) -> dict[str, float]:
    """Compute key radius metrics from simulation result."""
    R = result.radius
    R0 = result.eom.R0
    max_R = float(jnp.max(R))
    min_R = float(jnp.min(R))
    return {
        "max_radius": max_R,
        "min_radius": min_R,
        "max_ratio": max_R / R0,
        "min_ratio": R0 / min_R,
        "swing_ratio": max_R / min_R,
    }


@dataclass
class PlotArrays:
    """Convenient numpy arrays for plotting a simulation."""

    time_us: np.ndarray
    radius_um: np.ndarray
    pressure_kpa: np.ndarray


def arrays_from_result(result: SimulationResult) -> PlotArrays:
    """Convert simulation result to plottable numpy arrays in convenient units."""
    return PlotArrays(
        time_us=np.asarray(result.ts) / 1e-6,
        radius_um=np.asarray(result.radius) / 1e-6,
        pressure_kpa=np.asarray(result.driving_pressure) / 1e3,
    )
