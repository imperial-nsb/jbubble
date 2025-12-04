"""Diffrax-based Rayleigh-Plesset solver."""

from __future__ import annotations

from typing import Any, Tuple

import diffrax
import jax
import jax.numpy as jnp
import equinox as eqx

from .bubble import Bubble
from .pulse import Pulse

State = jax.Array
Args = Tuple[Bubble, Pulse]


def bubble_equation(t: Any, state: State, args: Args) -> State:
    R, R_dot = state
    bubble, pulse = args

    R0 = bubble.R0
    vdw = bubble.vdw
    gamma = bubble.gamma
    c_L = bubble.c_L
    P_amb = bubble.P_amb
    rho_L = bubble.rho_L
    mu_L = bubble.mu_L
    kappa_s = bubble.kappa_s
    sigma_R0 = bubble.sigma_R0

    sigma = bubble.surface_tension(R)
    P_drive = pulse(t)

    P_gas0 = P_amb + 2.0 * sigma_R0 / R0
    P_gas = P_gas0 * ((R0**3 - vdw**3) / (R**3 - vdw**3)) ** gamma

    P_surf = 2.0 * sigma / R
    P_visc = 4.0 * mu_L * R_dot / R
    P_surf_visc = 4.0 * kappa_s * R_dot / (R**2)

    damping_term = 1.0 - (3.0 * gamma * (R**3) * R_dot) / (c_L * (R**3 - vdw**3))

    forces = (P_gas * damping_term) - P_surf - P_visc - P_surf_visc - P_drive - P_amb
    R_ddot = (forces / rho_L - 1.5 * R_dot**2) / R

    return jnp.stack([R_dot, R_ddot])


class SaveSpec(eqx.Module):
    """Convenience wrapper for controlling solver outputs."""

    num_samples: int = eqx.field(default=1024, static=True)

    def build(self, t0: float, t1: float) -> diffrax.SaveAt:
        ts = jnp.linspace(t0, t1, self.num_samples)
        return diffrax.SaveAt(ts=ts)


def solve_bubble(
    bubble: Bubble,
    pulse: Pulse,
    *,
    t_span: Tuple[float, float] | None = None,
    dt0: float = 1e-3,
    save_spec: SaveSpec | None = None,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    progress: bool = False,
) -> diffrax.Solution:
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
    y0 = jnp.array([bubble.R0, 0.0])
    saveat = save_spec.build(t0, t1)
    term = diffrax.ODETerm(bubble_equation)
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
        max_steps=10_000,
        progress_meter=progress_meter,
        throw=False,
    )
