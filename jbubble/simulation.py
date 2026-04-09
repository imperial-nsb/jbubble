"""High-level helpers for running bubble dynamics simulations."""

from __future__ import annotations

import warnings
from typing import Any, cast

import diffrax
import equinox as eqx
import jax
from jax.typing import ArrayLike

from .bubble.eom import EquationOfMotion
from .bubble.state import BubbleState, ConfinedBubbleState
from .pulse import Pulse
from .solver import SaveSpec, SolverConfig, solve_eom


class SimulationResult(eqx.Module):
    """Output of :func:`run_simulation`.

    Generalises over any :class:`~jbubble.bubble.BubbleState` subclass:
    adding new degrees of freedom (e.g. temperature) to the state requires
    no changes here.

    All array quantities are in SI units, sampled at ``ts``.

    Attributes
    ----------
    ts : jax.Array, shape (N,)
        Time points [s].
    state : BubbleState, each field shape (N,)
        Full state trajectory.  ``state.R`` is the bubble radius,
        ``state.R_dot`` the radial velocity.  For confined models
        (``ConfinedBubbleState``) also carries ``state.a`` and ``state.a_dot``.
    state_dot : BubbleState, each field shape (N,)
        Time-derivative trajectory d(state)/dt returned by the EoM.
        ``state_dot.R_dot`` is the radial acceleration R̈(t).
    driving_pressure : jax.Array, shape (N,)
        Applied acoustic pressure at the bubble [Pa].
    converged : jax.Array
        Boolean scalar; ``True`` if the ODE solver converged successfully.
    """

    ts: jax.Array
    state: BubbleState
    state_dot: BubbleState
    driving_pressure: jax.Array
    converged: jax.Array

    # ── convenience accessors ──────────────────────────────────────────────────

    @property
    def radius(self) -> jax.Array:
        """Bubble wall radius R(t) [m]."""
        return self.state.R

    @property
    def radial_velocity(self) -> jax.Array:
        """Bubble wall velocity Ṙ(t) [m/s]."""
        return self.state.R_dot

    @property
    def radial_acceleration(self) -> jax.Array:
        """Bubble wall acceleration R̈(t) [m/s²], evaluated from the EoM RHS."""
        return self.state_dot.R_dot

    @property
    def has_vessel(self) -> bool:
        """``True`` for confined-bubble models (``ConfinedBubbleState``)."""
        return isinstance(self.state, ConfinedBubbleState)

    @property
    def vessel_radius(self) -> jax.Array | None:
        """Vessel wall radius a(t) [m], or ``None`` for unconfined models."""
        return self.state.a if self.has_vessel else None  # ty: ignore[unresolved-attribute]

    @property
    def vessel_velocity(self) -> jax.Array | None:
        """Vessel wall velocity ȧ(t) [m/s], or ``None`` for unconfined models."""
        return self.state.a_dot if self.has_vessel else None  # ty: ignore[unresolved-attribute]


def run_simulation(
    eom: EquationOfMotion,
    pulse: Pulse,
    *,
    save_spec: SaveSpec,
    state0: Any = None,
    t_max: ArrayLike | None = None,
    config: SolverConfig | None = None,
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
    t_max : float, optional
        Integration end time [s].  ``None`` uses ``pulse.t_end``.
    config : SolverConfig, optional
        ODE solver settings.
    progress : bool
        Show progress meter.

    Returns
    -------
    SimulationResult
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
