"""Diffrax-based solver for multiple bubble models using multibubble.py."""

from typing import Any, Tuple

import diffrax
import jax
import jax.numpy as jnp
import equinox as eqx
from .pulse import Pulse
from .bubble import BubbleBase
jax.config.update("jax_enable_x64", True)

State = jax.Array

class SaveSpec(eqx.Module):
    """Convenience wrapper for controlling solver outputs."""

    num_samples: int = eqx.field(default=1024, static=True)

    def build(self, t0: float, t1: float) -> diffrax.SaveAt:
        ts = jnp.linspace(t0, t1, self.num_samples)
        return diffrax.SaveAt(ts=ts)


def solve_bubble(
    bubble: BubbleBase,
    pulse: Pulse,
    *,
    t_span: Tuple[float, float] | None = None,
    dt0: float = 1e-3,
    save_spec: SaveSpec | None = None,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    progress: bool = False,
    max_steps: int = 10_000,
) -> diffrax.Solution:
    """
    Solve the bubble dynamics for any bubble model with a given pulse.
    
    Parameters
    ----------
    bubble : BubbleBase
        Bubble model instance (e.g., MarmottantBubble, MarmottantGompertz)
        Must have bubble_equation(t, state, pulse) method
    pulse : Pulse
        Driving pulse
    t_span : Tuple[float, float], optional
        Time span (t0, t1). If None, computed from pulse duration
    dt0 : float
        Initial time step
    save_spec : SaveSpec, optional
        Output specification. Default: 1000 samples
    solver : diffrax.AbstractSolver, optional
        ODE solver. Default: Kvaerno5
    stepsize_controller : diffrax.AbstractStepSizeController, optional
        Step size controller. Default: PIDController(rtol=1e-3, atol=1e-6)
    progress : bool
        Show progress meter
    max_steps : int
        Maximum number of steps
    
    Returns
    -------
    diffrax.Solution
        Solution object with ts, ys (radius and velocity over time)
    """
    if t_span is None:
        pulse_duration = pulse.cycle_num / pulse.freq
        t_span = (0.0, pulse.initial_time + 2.0 * pulse_duration)

    if save_spec is None:
        save_spec = SaveSpec(num_samples=1000)

    if solver is None:
        solver = diffrax.Kvaerno5()

    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

    t0, t1 = t_span
    y0 = bubble.initial_state()
    saveat = save_spec.build(t0, t1)
    
    # Create ODE term that calls bubble.bubble_equation
    def ode_func(t, state, args):
        bubble_model, pulse_model = args
        return bubble_model.bubble_equation(t, state, pulse_model)
    
    term = diffrax.ODETerm(ode_func)
    progress_meter = diffrax.TextProgressMeter() if progress else diffrax.NoProgressMeter()

    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=(bubble, pulse),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        progress_meter=progress_meter,
        throw=False,
    )
