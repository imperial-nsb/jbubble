"""High-level helpers for running and post-processing simulations with multiple bubble models."""

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
    """Results from a bubble simulation."""
    ts: jax.Array
    ys: jax.Array           # shape (T, 3) normally, (T, 5) with vessel
                            # index 0: R,   1: Ṙ,   2: R̈
                            # index 3: R_v, 4: Ṙ_v  (vessel models only)
    driving_pressure: jax.Array
    converged: jax.Array
    bubble: Bubble
    pulse: Pulse
    units: Units

    @property
    def radius(self) -> jax.Array:
        """Bubble radius over time."""
        return self.ys[..., 0]

    @property
    def radial_velocity(self) -> jax.Array:
        """Radial velocity over time."""
        return self.ys[..., 1]

    @property
    def radial_acceleration(self) -> jax.Array:
        """Analytical R̈, computed from the ODE right-hand side at solve time."""
        return self.ys[..., 2]

    def radiated_pressure(self, d: float) -> jax.Array:
        """Far-field radiated pressure at sensor distance *d* [m].

        Uses the acoustic monopole approximation (Leighton 1994)::

            P_rad = (ρ R / d) (R R̈ + 2 Ṁ²) − (ρ/4) Ṁ² (R/d)⁴

        Parameters
        ----------
        d : float
            Distance from the bubble centre to the sensor [m].
        """
        R     = self.radius
        Rdot  = self.radial_velocity
        Rddot = self.radial_acceleration
        rho   = self.bubble.rho_L
        term1 = (rho * R / d) * (R * Rddot + 2.0 * Rdot**2)
        term2 = (rho / 4.0) * Rdot**2 * (R / d)**4
        return term1 - term2

    @property
    def has_vessel(self) -> bool:
        return self.ys.shape[-1] >= 5

    @property
    def vessel_radius(self) -> jax.Array | None:
        if self.has_vessel:
            return self.ys[..., 3]
        return None

    @property
    def vessel_velocity(self) -> jax.Array | None:
        if self.has_vessel:
            return self.ys[..., 4]
        return None


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

    assert sol.ts is not None and sol.ys is not None

    ts = sol.ts * units.T_scale
    ys = bubble.rescale_state(sol.ys, units)   # (T, 2) or (T, 4) for vessel
    driving_pressure = jax.vmap(scaled_pulse)(sol.ts) * units.P_scale

    # Compute R̈ analytically from the ODE RHS, then splice it in at index 2
    # so the ys layout is [R, Ṙ, R̈, (R_v, Ṙ_v)] — indices always match.
    def _rddot(t, state):
        return scaled_bubble.bubble_equation(t, state, scaled_pulse)[1]
    rddot = jax.vmap(_rddot)(sol.ts, sol.ys) * units.acc_scale  # (T,)
    ys = jnp.concatenate([ys[..., :2], rddot[..., None], ys[..., 2:]], axis=-1)

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
