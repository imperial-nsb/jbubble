"""High-level helpers for running and post-processing simulations."""

from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax
import numpy as np

from .bubble import Bubble
from .pulse import Pulse
from .solver import SaveSpec, solve_bubble
from .units import Units


class SimulationResult(eqx.Module):
    ts: jax.Array
    ys: jax.Array
    driving_pressure: jax.Array
    converged: jax.Array
    bubble: Bubble
    pulse: Pulse
    units: Units

    @property
    def radius(self) -> jax.Array:
        return self.ys[..., 0]

    @property
    def radial_velocity(self) -> jax.Array:
        return self.ys[..., 1]



def run_simulation(
    bubble: Bubble,
    pulse: Pulse,
    *,
    units: Units,
    save_spec: SaveSpec,
    window_s: float = 20e-6, # [s]
    dt0: float = 1e-3,
    max_steps: int = 10_000,
    progress: bool = False,
) -> SimulationResult:
    """
    Scale, solve, rescale.
    """
    scaled_bubble = bubble.get_scaled(units)
    scaled_pulse = pulse.get_scaled(units)
    scaled_t_span = (0.0, window_s / units.T_scale)

    sol = solve_bubble(
        scaled_bubble,
        scaled_pulse,
        t_span=scaled_t_span,
        dt0=dt0,
        save_spec=save_spec,
        progress=progress,
        max_steps=max_steps,
    )

    assert sol.ts is not None and sol.ys is not None

    ts = sol.ts * units.T_scale
    radius = sol.ys[:, 0] * units.L_scale
    radial_velocity = sol.ys[:, 1] * units.vel_scale
    ys = jnp.stack([radius, radial_velocity], axis=-1)
    driving_pressure = jax.vmap(scaled_pulse)(sol.ts) * units.P_scale

    return SimulationResult(
        ts=ts,
        ys=ys,
        driving_pressure=driving_pressure,
        converged=diffrax.is_successful(sol.result),
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


@dataclass
class PlotArrays:
    """Convenient numpy arrays for plotting a simulation."""

    time_us: np.ndarray
    radius_um: np.ndarray
    pressure_kpa: np.ndarray


def arrays_from_result(result: SimulationResult) -> PlotArrays:
    units = result.units
    return PlotArrays(
        time_us=np.asarray(result.ts) / units.T_scale,
        radius_um=np.asarray(result.radius) / units.L_scale,
        pressure_kpa=np.asarray(result.driving_pressure) / units.P_scale,
    )


