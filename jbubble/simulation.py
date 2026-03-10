"""High-level helpers for running and post-processing bubble dynamics simulations."""

from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .bubble.interfaces import EquationOfMotion
from .pulse import Pulse
from .solver import SaveSpec, solve_eom
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
        The equation of motion used (physical-unit copy).
    pulse : Pulse
        The driving pulse used (physical-unit copy).
    units : Units
        Non-dimensionalisation factors used during the simulation.
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
    units: Units

    @property
    def has_vessel(self) -> bool:
        return self.vessel_radius is not None

    def radiated_pressure(self, d: float) -> jax.Array:
        """Far-field radiated pressure at sensor distance *d* [m].

        Uses the acoustic monopole approximation (Leighton 1994)::

            P_rad = (rho R / d) (R Rddot + 2 Rdot^2) - (rho/4) Rdot^2 (R/d)^4

        Parameters
        ----------
        d : float
            Distance from the bubble centre to the sensor [m].
        """
        R = self.radius
        Rdot = self.radial_velocity
        Rddot = self.radial_acceleration
        rho = self.eom.rho_L
        term1 = (rho * R / d) * (R * Rddot + 2.0 * Rdot**2)
        term2 = (rho / 4.0) * Rdot**2 * (R / d) ** 4
        return term1 - term2


def run_simulation(
    eom: EquationOfMotion,
    pulse: Pulse,
    *,
    units: Units,
    save_spec: SaveSpec,
    window_s: float = 20e-6,
    dt0: float = 1e-3,
    max_steps: int = 10_000,
    progress: bool = False,
) -> SimulationResult:
    """Run a simulation: scale EoM and pulse, solve ODE, rescale results.

    Parameters
    ----------
    eom : EquationOfMotion
        Assembled equation of motion (e.g. ``KellerMiksis``).
    pulse : Pulse
        Driving pulse.
    units : Units
        Unit scaling object.
    save_spec : SaveSpec
        Output specification (number of samples).
    window_s : float
        Simulation time window in seconds.
    dt0 : float
        Initial time step.
    max_steps : int
        Maximum ODE steps.
    progress : bool
        Show progress meter.

    Returns
    -------
    SimulationResult
        Scaled results with time, radius, velocity, pressure, convergence info.
    """
    scaled_eom = eom.get_scaled(units)
    scaled_pulse = pulse.get_scaled(units)
    scaled_t_span = (0.0, window_s / units.T_scale)

    sol = solve_eom(
        scaled_eom,
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
    ys = eom.rescale_state(sol.ys, units)
    driving_pressure = jax.vmap(scaled_pulse)(sol.ts) * units.P_scale

    # Compute acceleration analytically from the ODE RHS.
    def _rddot(t, state):
        return scaled_eom(t, state, scaled_pulse)[1]

    rddot = jax.vmap(_rddot)(sol.ts, sol.ys) * units.acc_scale

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
        eom=eom,
        pulse=pulse,
        units=units,
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
    units = result.units
    return PlotArrays(
        time_us=np.asarray(result.ts) / units.T_scale,
        radius_um=np.asarray(result.radius) / units.L_scale,
        pressure_kpa=np.asarray(result.driving_pressure) / units.P_scale,
    )
