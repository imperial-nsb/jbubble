"""Acoustic emission models for bubble dynamics.

Each model computes the radiated acoustic pressure at a field point from a
solved bubble trajectory, at a different level of physical fidelity:
[`IncompressibleMonopole`][jbubble.acoustics.emission.IncompressibleMonopole]
assumes an incompressible liquid, and
[`QuasiAcoustic`][jbubble.acoustics.emission.QuasiAcoustic] adds a
retarded-time correction.

Examples
--------
```python
from jbubble.acoustics import IncompressibleMonopole

emission = IncompressibleMonopole(rho_L=998.0)
p_rad = emission(result, r=0.01)  # at 1 cm
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
    series.

    To evaluate several field-point distances, use `jax.vmap`:

    ```python
    distances = jnp.array([0.001, 0.005, 0.01])
    p_all = jax.vmap(lambda r: model(result, r))(distances)
    # shape (3, N)
    ```
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
            Radiated pressure [Pa] at each saved time point.
        """
        ...


class IncompressibleMonopole(EmissionModel):
    r"""Incompressible monopole radiation.

    Assumes an incompressible surrounding liquid, so the time derivative
    of the volume flux gives the radiated pressure at distance $r$:

    $$
    p_\text{rad}(r, t) = \frac{\rho_L}{r}\frac{\mathrm{d}}{\mathrm{d}t}\left(R^2\dot{R}\right)
        = \frac{\rho_L}{r}\left(2R\dot{R}^2 + R^2\ddot{R}\right)
    $$

    This is the simplest acoustic emission model. It's accurate when the
    bubble-wall Mach number $M = \dot{R}/c_L \ll 1$ and the field point
    is in the geometric near field ($r \ll c_L/f$).

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
        R = result.state.R
        R_dot = result.state.R_dot
        R_ddot = result.state_dot.R_dot
        return (
            jnp.asarray(self.rho_L)
            / jnp.asarray(r)
            * (2.0 * R * R_dot**2 + R**2 * R_ddot)
        )


class QuasiAcoustic(EmissionModel):
    r"""Quasi-acoustic emission with a retarded-time correction.

    Accounts for the finite speed of sound by evaluating the bubble-wall
    quantities at the retarded time $t_\text{ret} = t - r/c_L$:

    $$
    p_\text{rad}(r, t) = \frac{\rho_L R^2(t_\text{ret})}{r}
        \left[\ddot{R}(t_\text{ret}) + \frac{2\dot{R}^2(t_\text{ret})}{R(t_\text{ret})}\right]
    $$

    The model evaluates the trajectory at retarded times with linear
    interpolation (`jnp.interp`). Where $t_\text{ret}$ falls before the
    first saved time, `jnp.interp` clamps the values to the first saved
    (initial) state. That's physically reasonable, because the bubble is
    quiescent before excitation.

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
        delay = r / self.c_L
        t_ret = result.ts - delay

        # Interpolate bubble-wall quantities at retarded times.
        R_ret = jnp.interp(t_ret, result.ts, result.state.R)
        R_dot_ret = jnp.interp(t_ret, result.ts, result.state.R_dot)
        R_ddot_ret = jnp.interp(t_ret, result.ts, result.state_dot.R_dot)

        return (
            jnp.asarray(self.rho_L)
            * R_ret**2
            / jnp.asarray(r)
            * (R_ddot_ret + 2.0 * R_dot_ret**2 / R_ret)
        )
