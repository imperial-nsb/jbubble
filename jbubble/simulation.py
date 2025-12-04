"""High-level helpers for running and post-processing simulations."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax

from .bubble import Bubble
from .pulse import Pulse
from .shapes import DEFAULT_PULSE_LIBRARY
from .solver import SaveSpec, solve_bubble
from .units import Units


@dataclass
class SimulationResult:
    ts: jax.Array
    ys: jax.Array
    driving_pressure: jax.Array
    converged: bool
    bubble: Bubble
    pulse: Pulse
    units: Units

    @property
    def radius(self) -> jax.Array:
        return self.ys[:, 0]

    @property
    def radial_velocity(self) -> jax.Array:
        return self.ys[:, 1]


def build_pulse(shape: str, **kwargs) -> Pulse:
    shape_func = DEFAULT_PULSE_LIBRARY.get(shape, DEFAULT_PULSE_LIBRARY["sine"])
    return Pulse(shape_func=shape_func, **kwargs)


def default_bubble(R0: float = 4e-6) -> Bubble:
    return Bubble(
        R0=R0,
        R_buckle=0.99 * R0,
        gamma=1.07,
        chi=0.38,
        mu_L=0.00089,
        kappa_s=2.4e-9,
        rho_L=1000.0,
        c_L=1498.0,
        P_amb=101.3e3,
        sigma_L=72e-3,
    )


def default_pulse(freq: float = 800e3, pressure: float = 1e6) -> Pulse:
    return build_pulse(
        "sine",
        freq=freq,
        pressure=pressure,
        cycle_num=10,
        initial_time=1e-6,
        n=3,
        apply_hann=False,
    )


def run_simulation(
    bubble: Bubble | None = None,
    pulse: Pulse | None = None,
    *,
    units: Units | None = None,
    save_spec: SaveSpec | None = None,
    dt0: float = 1e-3,
    progress: bool = False,
) -> SimulationResult:
    """Solve once in SI units.

    For differentiable workflows, call :func:`jbubble.solver.solve_bubble`
    directly (optionally wrapped in :func:`equinox.filter_jit`) and reuse the
    returned function handle inside optimisation loops.
    """
    units = units or Units()
    bubble = bubble or default_bubble()
    pulse = pulse or default_pulse()

    scaled_bubble = bubble.get_scaled(units)
    scaled_pulse = pulse.get_scaled(units)

    sol = solve_bubble(
        scaled_bubble,
        scaled_pulse,
        dt0=dt0,
        save_spec=save_spec or SaveSpec(1000),
        progress=progress,
    )

    if sol.ts is None or sol.ys is None:
        raise ValueError("SaveAt(ts=...) is required for visualisation in run_simulation")

    ts = sol.ts * units.T_scale
    radius = sol.ys[:, 0] * units.L_scale
    radial_velocity = sol.ys[:, 1] * units.vel_scale
    ys = jnp.stack([radius, radial_velocity], axis=-1)
    driving_pressure = jax.vmap(scaled_pulse)(sol.ts) * units.P_scale
    converged = bool(sol.result == diffrax.RESULTS.successful) if hasattr(sol, "result") else True

    return SimulationResult(
        ts=ts,
        ys=ys,
        driving_pressure=driving_pressure,
        converged=converged,
        bubble=bubble,
        pulse=pulse,
        units=units,
    )


def compute_radius_metrics(result: SimulationResult) -> dict[str, float]:
    R = result.radius
    R0 = result.bubble.R0
    max_R = float(jnp.max(R))
    min_R = float(jnp.min(R))
    return {
        "max_radius": max_R,
        "min_radius": min_R,
        "max_ratio": max_R / R0,
        "min_ratio": R0 / min_R,
        "swing_ratio": max_R / min_R,
    }
