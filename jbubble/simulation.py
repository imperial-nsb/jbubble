"""High-level helpers for running bubble dynamics simulations."""

from __future__ import annotations

import warnings
from typing import Any, cast

import diffrax
import equinox as eqx
import jax
from jax.typing import ArrayLike

from .bubble.eom import EquationOfMotion
from .bubble.state import BubbleState
from .pulse import Pulse
from .solver import SaveSpec, SolverConfig, solve_eom

__all__ = ["SimulationResult", "run_simulation"]


class SimulationResult(eqx.Module):
    r"""Output of [`run_simulation`][jbubble.simulation.run_simulation].

    Generalises over any [`BubbleState`][jbubble.bubble.state.BubbleState]
    subclass: when you add degrees of freedom, such as temperature, to the
    state, this class needs no changes.

    All array quantities are in SI units, sampled at `ts`.

    Attributes
    ----------
    ts : jax.Array, shape (N,)
        Time points [s].
    state : BubbleState, each field shape (N,)
        Full state trajectory. `state.R` is the bubble radius and
        `state.R_dot` the radial velocity.
    state_dot : BubbleState, each field shape (N,)
        Time-derivative trajectory, $\mathrm{d}(\text{state})/\mathrm{d}t$,
        that the equation of motion returns. `state_dot.R_dot` is the
        radial acceleration $\ddot{R}(t)$.
    driving_pressure : jax.Array, shape (N,)
        Applied acoustic pressure at the bubble [Pa].
    converged : jax.Array
        Boolean scalar: `True` if the ODE solver converged successfully.
    """

    ts: jax.Array
    state: BubbleState
    state_dot: BubbleState
    driving_pressure: jax.Array
    converged: jax.Array

    # ── convenience accessors ──────────────────────────────────────────────────

    @property
    def radius(self) -> jax.Array:
        """Bubble wall radius $R(t)$ [m]."""
        return self.state.R

    @property
    def radial_velocity(self) -> jax.Array:
        r"""Bubble wall velocity $\dot{R}(t)$ [m/s]."""
        return self.state.R_dot

    @property
    def radial_acceleration(self) -> jax.Array:
        r"""Bubble wall acceleration $\ddot{R}(t)$ [m/s²], from the equation of motion."""
        return self.state_dot.R_dot


def run_simulation(
    eom: EquationOfMotion,
    pulse: Pulse,
    *,
    save_spec: SaveSpec | None = None,
    state0: Any = None,
    t_max: ArrayLike | None = None,
    config: SolverConfig | None = None,
    progress: bool = False,
) -> SimulationResult:
    """Run a simulation and return the results in SI units.

    Parameters
    ----------
    eom : EquationOfMotion
        Assembled equation of motion, such as
        [`KellerMiksis`][jbubble.bubble.eom.KellerMiksis].
    pulse : Pulse
        Driving pulse.
    save_spec : SaveSpec, optional
        Output specification (number of samples). `None` uses
        [`SaveSpec()`][jbubble.solver.SaveSpec], 1024 evenly spaced time
        points.
    state0 : BubbleState, optional
        Initial state. `None` uses `eom.initial_state()`.
    t_max : float, optional
        Integration end time [s]. `None` uses `pulse.t_end`.
    config : SolverConfig, optional
        ODE solver settings. `None` uses
        [`SolverConfig()`][jbubble.solver.SolverConfig].
    progress : bool
        Whether to show a text progress meter. Default: `False`.

    Returns
    -------
    SimulationResult
        Trajectory, time derivatives, driving pressure, and convergence
        flag.
    """
    sol = solve_eom(
        eom,
        pulse,
        y0=state0,  # None → solve_eom calls eom.initial_state()
        t_max=t_max,
        save_spec=save_spec,
        config=config,
        progress=progress,
    )

    ts = cast(jax.Array, sol.ts)
    ys = cast(BubbleState, sol.ys)
    ys_dot: BubbleState = jax.vmap(lambda t, x: eom(t, x, pulse))(ts, ys)

    converged = diffrax.is_successful(sol.result)

    def _warn_not_converged(c: bool) -> None:
        if not c:
            warnings.warn(
                "ODE solver did not converge. "
                "Returned trajectory may be incomplete. Check `result.converged` before use.",
                UserWarning,
                stacklevel=2,
            )

    jax.debug.callback(_warn_not_converged, converged)

    return SimulationResult(
        ts=ts,
        state=ys,
        state_dot=ys_dot,
        driving_pressure=jax.vmap(pulse)(ts),
        converged=converged,
    )
