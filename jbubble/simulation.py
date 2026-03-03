"""High-level helpers for running and post-processing simulations with multiple bubble models."""

from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .bubble import Bubble
from .pulse import Pulse
from .solver import SaveSpec, solve_bubble
from .units import Units


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
        Bubble wall velocity Ṙ(t) [m/s].
    radial_acceleration : jax.Array, shape (N,)
        Bubble wall acceleration R̈(t) [m/s²], computed analytically from the
        ODE right-hand side (not numerical differentiation).
    vessel_radius : jax.Array or None, shape (N,)
        Vessel wall radius Rᵥ(t) [m]. Non-``None`` only for
        :class:`~jbubble.bubble.SphericalConfinement`.
    vessel_velocity : jax.Array or None, shape (N,)
        Vessel wall velocity Ṙᵥ(t) [m/s]. Non-``None`` only for
        :class:`~jbubble.bubble.SphericalConfinement`.
    driving_pressure : jax.Array, shape (N,)
        Applied acoustic pressure at the bubble [Pa].
    converged : jax.Array
        Boolean scalar; ``True`` if the ODE solver converged successfully.
    bubble : Bubble
        The bubble model used (physical-unit copy).
    pulse : Pulse
        The driving pulse used (physical-unit copy).
    units : Units
        Non-dimensionalisation factors used during the simulation.
    """

    ts: jax.Array
    radius: jax.Array  # R    [m]
    radial_velocity: jax.Array  # Ṙ   [m/s]
    radial_acceleration: jax.Array  # R̈  [m/s²], computed analytically from ODE RHS
    vessel_radius: jax.Array | None  # R_v  [m]   (vessel models only)
    vessel_velocity: jax.Array | None  # Ṙ_v [m/s]  (vessel models only)
    driving_pressure: jax.Array
    converged: jax.Array
    bubble: Bubble
    pulse: Pulse
    units: Units

    @property
    def has_vessel(self) -> bool:
        return self.vessel_radius is not None

    def radiated_pressure(self, d: float) -> jax.Array:
        """Far-field radiated pressure at sensor distance *d* [m].

        Uses the acoustic monopole approximation (Leighton 1994)::

            P_rad = (ρ R / d) (R R̈ + 2 Ṁ²) − (ρ/4) Ṁ² (R/d)⁴

        Parameters
        ----------
        d : float
            Distance from the bubble centre to the sensor [m].
        """
        R = self.radius
        Rdot = self.radial_velocity
        Rddot = self.radial_acceleration
        rho = self.bubble.rho_L  # type: ignore
        term1 = (rho * R / d) * (R * Rddot + 2.0 * Rdot**2)
        term2 = (rho / 4.0) * Rdot**2 * (R / d) ** 4
        return term1 - term2


def run_simulation(
    bubble: Bubble,
    pulse: Pulse,
    *,
    units: Units,
    save_spec: SaveSpec,
    window_s: float = 20e-6,  # [s]
    dt0: float = 1e-3,
    max_steps: int = 10_000,
    progress: bool = False,
) -> SimulationResult:
    """
    Run a simulation: scale bubble and pulse, solve ODE, rescale results.

    Parameters
    ----------
    bubble : BubbleBase
        Bubble model instance (e.g., MarmottantBubble, MarmottantGompertz)
    pulse : Pulse
        Driving pulse
    units : Units
        Unit scaling object
    save_spec : SaveSpec
        Output specification (number of samples)
    window_s : float
        Simulation time window in seconds
    dt0 : float
        Initial time step
    max_steps : int
        Maximum ODE steps
    progress : bool
        Show progress meter

    Returns
    -------
    SimulationResult
        Scaled results with time, radius, velocity, pressure, convergence info
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

    assert sol.ts is not None
    assert sol.ys is not None
    ts = sol.ts * units.T_scale
    ys = bubble.rescale_state(sol.ys, units)  # (T, 2) or (T, 4) for vessel
    driving_pressure = jax.vmap(scaled_pulse)(sol.ts) * units.P_scale

    # Compute acceleration analytically from the ODE RHS.
    def _rddot(t, state):
        return scaled_bubble.bubble_equation(t, state, scaled_pulse)[1]

    rddot = jax.vmap(_rddot)(sol.ts, sol.ys) * units.acc_scale  # (T,)

    has_vessel = ys.shape[-1] >= 4  # shape is static, safe outside jit

    return SimulationResult(
        ts=ts,
        radius=ys[..., 0],
        radial_velocity=ys[..., 1],
        radial_acceleration=rddot,
        vessel_radius=ys[..., 2] if has_vessel else None,
        vessel_velocity=ys[..., 3] if has_vessel else None,
        driving_pressure=driving_pressure,
        converged=diffrax.is_successful(sol.result),
        bubble=bubble,
        pulse=pulse,
        units=units,
    )


def compute_radius_metrics(result: SimulationResult) -> dict[str, float]:
    """Compute key radius metrics from simulation result."""
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
    """Convert simulation result to plottable numpy arrays in convenient units."""
    units = result.units
    return PlotArrays(
        time_us=np.asarray(result.ts) / units.T_scale,
        radius_um=np.asarray(result.radius) / units.L_scale,
        pressure_kpa=np.asarray(result.driving_pressure) / units.P_scale,
    )
