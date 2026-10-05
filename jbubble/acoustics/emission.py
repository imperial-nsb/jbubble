"""Acoustic emission models for bubble dynamics.

Each model computes the radiated acoustic pressure at a field point from a
solved bubble trajectory, at a different level of physical fidelity:
[`IncompressibleMonopole`][jbubble.acoustics.emission.IncompressibleMonopole]
assumes an incompressible liquid, and
[`QuasiAcoustic`][jbubble.acoustics.emission.QuasiAcoustic] adds the
propagation delay $r/c_L$.

Every model returns the pressure series together with the time at which
the field point receives each sample,
[`observer_time`][jbubble.acoustics.emission.EmissionModel.observer_time].

Examples
--------
```python
from jbubble.acoustics import QuasiAcoustic

emission = QuasiAcoustic(rho_L=998.0, c_L=1500.0)
p_rad = emission(result, r=0.01)  # at 1 cm
t_obs = emission.observer_time(result, r=0.01)  # result.ts + r / c_L
```
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..simulation import SimulationResult

__all__ = ["EmissionModel", "IncompressibleMonopole", "QuasiAcoustic"]


class EmissionModel(eqx.Module, abc.ABC):
    """Acoustic emission model: bubble trajectory → radiated pressure.

    Subclasses implement `__call__`, which takes a solved
    [`SimulationResult`][jbubble.simulation.SimulationResult] and a
    field-point distance `r`, and returns the radiated pressure time
    series. Sample `i` of that series reaches the field point at
    `observer_time(result, r)[i]`.

    To evaluate several field-point distances, use `jax.vmap`:

    ```python
    distances = jnp.array([0.001, 0.005, 0.01])
    p_all = jax.vmap(lambda r: model(result, r))(distances)
    # shape (3, N)
    ```

    Notes
    -----
    The radiated pressure of an inertial collapse is a very short spike.
    For a 2 µm air bubble driven at 300 kPa and 1 MHz, the collapse peak
    has a full width at half maximum of about 0.4 ns, and the default
    [`SaveSpec`][jbubble.solver.SaveSpec], 1024 samples over the 10 µs of
    a 5-cycle pulse, captures only about a tenth of its height. To resolve
    such peaks, sample the trajectory every 0.1 ns or more finely, for
    example with `SaveSpec(num_samples=100_001)` over 10 µs. The solver
    accuracy isn't the limit here: the sampling is.
    """

    @abc.abstractmethod
    def __call__(
        self,
        result: SimulationResult,
        r: ArrayLike,
    ) -> jax.Array:
        """Compute the radiated pressure at distance `r`.

        Parameters
        ----------
        result : SimulationResult
            Solved bubble trajectory (`state`, `state_dot`, `ts`).
        r : float or jax.Array
            Distance from the bubble centre to the field point [m].

        Returns
        -------
        jax.Array, shape (N,)
            Radiated pressure [Pa]. Sample `i` reaches the field point at
            `observer_time(result, r)[i]`.
        """
        ...

    def observer_time(self, result: SimulationResult, r: ArrayLike) -> jax.Array:
        """Return the time at which each emitted sample reaches distance `r`.

        The base implementation assumes instantaneous propagation, so it
        returns `result.ts`.

        Parameters
        ----------
        result : SimulationResult
            Solved bubble trajectory.
        r : float or jax.Array
            Distance from the bubble centre to the field point [m].

        Returns
        -------
        jax.Array, shape (N,)
            Observer times [s].
        """
        del r
        return result.ts


def _monopole(result: SimulationResult, rho_L: ArrayLike, r: ArrayLike) -> jax.Array:
    r"""Return $\rho_L (2R\dot{R}^2 + R^2\ddot{R}) / r$ at each saved sample."""
    R = result.state.R
    R_dot = result.state.R_dot
    R_ddot = result.state_dot.R_dot
    return jnp.asarray(rho_L) / jnp.asarray(r) * (2.0 * R * R_dot**2 + R**2 * R_ddot)


class IncompressibleMonopole(EmissionModel):
    r"""Incompressible monopole radiation.

    Assumes an incompressible surrounding liquid, so the volume
    acceleration $\ddot{V}$ of the bubble, $V = 4\pi R^3/3$, gives the
    radiated pressure at distance $r$:

    $$
    p_\text{rad}(r, t) = \frac{\rho_L \ddot{V}}{4\pi r}
        = \frac{\rho_L}{r}\frac{\mathrm{d}}{\mathrm{d}t}\left(R^2\dot{R}\right)
        = \frac{\rho_L}{r}\left(2R\dot{R}^2 + R^2\ddot{R}\right)
    $$

    It's the far-field term of the incompressible pressure field
    $p - p_\infty = \rho_L\left[(R^2\ddot{R} + 2R\dot{R}^2)/r
    - R^4\dot{R}^2/(2r^4)\right]$, so it's accurate when:

    - the field point is far from the wall, $r \gg R$, where the dropped
      $R^4\dot{R}^2/(2r^4)$ term is negligible;
    - the field point is within a small fraction of a wavelength,
      $r \ll c_L/f$, so that the propagation delay doesn't matter;
    - the bubble-wall Mach number is small, $M = \dot{R}/c_L \ll 1$.

    Parameters
    ----------
    rho_L : float or jax.Array
        Liquid density [kg/m³].
    """

    rho_L: ArrayLike

    def __call__(
        self,
        result: SimulationResult,
        r: ArrayLike,
    ) -> jax.Array:
        return _monopole(result, self.rho_L, r)


class QuasiAcoustic(EmissionModel):
    r"""Monopole radiation delayed by the acoustic travel time.

    Accounts for the finite speed of sound: the field point at distance $r$
    receives the monopole pressure that the bubble emitted at the retarded
    time $t - r/c_L$:

    $$
    p_\text{rad}(r, t) = \frac{\rho_L}{r}
        \left(2R\dot{R}^2 + R^2\ddot{R}\right)\Big|_{t - r/c_L}
    $$

    The model returns the monopole series of
    [`IncompressibleMonopole`][jbubble.acoustics.emission.IncompressibleMonopole]
    on the shifted time axis
    [`observer_time`][jbubble.acoustics.emission.QuasiAcoustic.observer_time]
    `= result.ts + r / c_L`. The pressure values are therefore identical to
    the incompressible monopole's; only the arrival times differ. Because
    the model doesn't resample the trajectory, it keeps every collapse peak
    that the saved samples resolve. If you need the pressure on another time
    grid, interpolate the returned series, for example
    `jnp.interp(result.ts, t_obs, p_rad, left=p_rad[0])`, rather than the
    separate $R$, $\dot{R}$, and $\ddot{R}$ series: interpolating those and
    recombining them overshoots the true peak.

    The validity conditions of the monopole, $r \gg R$ and $M \ll 1$, still
    apply; the delay removes the $r \ll c_L/f$ condition.

    Parameters
    ----------
    rho_L : float or jax.Array
        Liquid density [kg/m³].
    c_L : float or jax.Array
        Speed of sound in the liquid [m/s].
    """

    rho_L: ArrayLike
    c_L: ArrayLike

    def __call__(
        self,
        result: SimulationResult,
        r: ArrayLike,
    ) -> jax.Array:
        return _monopole(result, self.rho_L, r)

    def observer_time(self, result: SimulationResult, r: ArrayLike) -> jax.Array:
        """Return `result.ts + r / c_L`, the arrival time of each sample [s].

        Parameters
        ----------
        result : SimulationResult
            Solved bubble trajectory.
        r : float or jax.Array
            Distance from the bubble centre to the field point [m].

        Returns
        -------
        jax.Array, shape (N,)
            Observer times [s].
        """
        return result.ts + jnp.asarray(r) / jnp.asarray(self.c_L)
