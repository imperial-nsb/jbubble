"""
Abstract base interfaces for modular bubble dynamics.

The liquid-side boundary pressure for a bubble is assembled from three
independent physical components:

    p_L(R, Rdot) = p_gas(R) - p_shell(R, Rdot) - p_medium(R, Rdot)

This module defines the four abstract component types that compose
into a full bubble dynamics model:

- ``GasModel``              — internal gas pressure  p_gas(R)
- ``SurfaceTensionModel``   — surface tension  sigma(R)
- ``ShellModel``            — shell/coating stress   p_shell(R, Rdot)
- ``MediumModel``           — surrounding medium     p_medium(R, Rdot)
- ``EquationOfMotion``      — macroscopic EoM assembling components into an ODE
"""

import abc
import dataclasses
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

# Type alias for the ODE state vector (typically [R, Rdot]).
State = jax.Array

# ---------------------------------------------------------------------------
# Field-scaling registry
# ---------------------------------------------------------------------------
# Maps field names to their corresponding ``Units`` attribute.  Used by
# ``_scale_module()`` to non-dimensionalise constructor parameters.

_FIELD_SCALES: dict[str, str] = {
    # Lengths
    "R0": "L_scale",
    "h": "L_scale",
    "d_s": "L_scale",
    "R_buckle": "L_scale",
    "tube_radius": "L_scale",
    "tube_length": "L_scale",
    "vessel_radius": "L_scale",
    "vessel_d": "L_scale",
    "tissue_d": "L_scale",
    # Pressures / elastic moduli
    "P_amb": "P_scale",
    "P_gas0": "P_scale",
    "G": "P_scale",
    "G_s": "P_scale",
    "vessel_E": "P_scale",
    # Surface tensions / shell elasticity
    "chi": "chi_scale",
    "sigma_L": "sigma_scale",
    "sigma_water": "sigma_scale",
    "sigma_break": "sigma_scale",
    # Dynamic viscosities
    "mu": "mu_scale",
    "mu_L": "mu_scale",
    "mu_s": "mu_scale",
    # Shell surface-dilatational viscosity
    "kappa_s": "kappa_scale",
    # Densities
    "rho_L": "rho_scale",
    "vessel_rho": "rho_scale",
    "tissue_rho": "rho_scale",
    # Velocities
    "c_L": "vel_scale",
    # Dimensionless
    "gamma": "unit_scale",
}


def _scale_module(module: eqx.Module, units: Any) -> eqx.Module:
    """Non-dimensionalise an Equinox module by scaling all dataclass fields.

    Sub-modules (GasModel, ShellModel, etc.) are recursively scaled.
    Scalar fields are divided by the appropriate unit scale from
    ``_FIELD_SCALES``.  Fields not in the registry are left unchanged.
    """
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(module):
        val = getattr(module, f.name)
        if isinstance(val, eqx.Module):
            kwargs[f.name] = _scale_module(val, units)
        elif f.name in _FIELD_SCALES:
            kwargs[f.name] = val / getattr(units, _FIELD_SCALES[f.name])
        else:
            kwargs[f.name] = val
    return type(module)(**kwargs)


class GasModel(eqx.Module, abc.ABC):
    """Internal gas pressure model.

    Computes the outward gas pressure p_gas as a function of the
    instantaneous radius R only.  The gas state is determined by bubble
    volume, so it has no dependence on wall velocity Rdot.

    Examples: polytropic law, van der Waals corrected gas.
    """

    @abc.abstractmethod
    def __call__(self, R: jax.Array) -> jax.Array:
        """Compute gas pressure p_gas(R).

        Parameters
        ----------
        R : scalar
            Current bubble radius.

        Returns
        -------
        scalar
            Gas pressure p_gas(R).
        """
        ...


class SurfaceTensionModel(eqx.Module, abc.ABC):
    """Surface tension law sigma(R).

    Defines how surface tension depends on the instantaneous radius.
    Separated from the shell mechanics so that any surface tension law
    can be paired with any shell model.

    Examples: constant (uncoated), piecewise Marmottant, smooth Gompertz.
    """

    @abc.abstractmethod
    def __call__(self, R: jax.Array) -> jax.Array:
        """Compute surface tension sigma(R).

        Parameters
        ----------
        R : scalar
            Current bubble radius.

        Returns
        -------
        scalar
            Surface tension sigma(R)  [N/m].
        """
        ...


class ShellModel(eqx.Module, abc.ABC):
    """Bubble shell / coating model.

    Computes the total inward stress from the shell, including:

    - Laplace pressure from surface tension:  2 sigma(R) / R
    - Shell viscous dissipation  (e.g. 4 kappa_s Rdot / R^2)
    - Shell elastic restoring forces  (for thick shells)

    Every ``ShellModel`` holds a ``SurfaceTensionModel`` as its ``sigma``
    field, separating the surface tension law from the shell mechanics.
    The ``surface_tension`` convenience method delegates to ``sigma(R)``
    and is used externally to compute equilibrium gas pressure
    P_gas0 = P_amb + 2 sigma(R0) / R0.
    """

    sigma: SurfaceTensionModel

    def surface_tension(self, R: jax.Array) -> jax.Array:
        """Compute surface tension sigma(R) by delegating to the inner model."""
        return self.sigma(R)

    @abc.abstractmethod
    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        """Compute total shell pressure p_shell(R, Rdot).

        This includes the Laplace pressure 2 sigma(R) / R plus any
        additional viscous or elastic shell terms.

        Parameters
        ----------
        R : scalar
            Current bubble radius.
        R_dot : scalar
            Current wall velocity.

        Returns
        -------
        scalar
            Total inward shell pressure.
        """
        ...


class MediumModel(eqx.Module, abc.ABC):
    """Surrounding medium (fluid / tissue) model.

    Computes the total inward viscous and elastic stresses exerted by the
    surrounding medium on the bubble wall.  For a Newtonian liquid this is
    just 4 mu Rdot / R.  Viscoelastic media (Kelvin-Voigt, Neo-Hookean, etc.)
    add elastic restoring forces.
    """

    @abc.abstractmethod
    def p_viscous(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        """Viscous contribution to the medium pressure."""
        ...

    @abc.abstractmethod
    def p_elastic(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        """Elastic contribution to the medium pressure."""
        ...

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        """Compute total medium pressure p_medium(R, Rdot).

        Parameters
        ----------
        R : scalar
            Current bubble radius.
        R_dot : scalar
            Current wall velocity.

        Returns
        -------
        scalar
            Total inward medium pressure (viscous + elastic).
        """
        return self.p_viscous(R, R_dot) + self.p_elastic(R, R_dot)


# ---------------------------------------------------------------------------
# Equation of Motion interface
# ---------------------------------------------------------------------------


class EquationOfMotion(eqx.Module, abc.ABC):
    """Macroscopic equation of motion for bubble dynamics.

    Assembles a ``GasModel``, ``ShellModel``, and ``MediumModel`` into a
    complete ODE right-hand side.  Concrete subclasses encode different
    EoM formulations (Rayleigh-Plesset, Keller-Miksis, ...) which differ
    in how they relate p_L to the radial acceleration Rddot.

    The liquid-side boundary pressure is computed by the concrete helper
    ``p_L`` as::

        p_L = gas(R) - shell(R, Rdot) - medium(R, Rdot)

    Because ``p_L`` is a regular method on an Equinox module, JAX can
    differentiate through it automatically.  EoMs that require dp_L/dt
    (e.g. Keller-Miksis) can use ``jax.grad(self.p_L, ...)`` instead of
    hand-coding analytical derivatives for every component combination.

    The driving acoustic pressure is received as a callable ``p_ac_fn``
    so that EoMs needing dp_ac/dt can compute it via ``jax.grad``.

    Fields
    ------
    gas : GasModel
        Internal gas pressure model.
    shell : ShellModel
        Shell / coating model.
    medium : MediumModel
        Surrounding medium model.
    R0 : float
        Equilibrium bubble radius  [m].
    P_amb : float
        Ambient (far-field) pressure  [Pa].
    rho_L : float
        Liquid density  [kg/m^3].  Used in the inertial terms of the EoM.
    """

    gas: GasModel
    shell: ShellModel
    medium: MediumModel
    R0: float
    P_amb: float
    rho_L: float

    # -- concrete helpers --------------------------------------------------

    def p_L(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        """Liquid-side boundary pressure.

        p_L = p_gas(R) - p_shell(R, Rdot) - p_medium(R, Rdot)
        """
        return self.gas(R) - self.shell(R, R_dot) - self.medium(R, R_dot)

    def initial_state(self) -> State:
        """Default initial state ``[R0, 0.0]``.

        Override for coupled systems with a larger state vector.
        """
        return jnp.array([self.R0, 0.0])

    def get_scaled(self, units: Any) -> "EquationOfMotion":
        """Return a dimensionless copy scaled by *units*.

        Recursively scales all sub-modules (gas, shell, medium) and
        scalar fields using ``_FIELD_SCALES``.
        """
        return _scale_module(self, units)

    def rescale_state(self, state: jax.Array, units: Any) -> jax.Array:
        """Rescale the state variables back to physical units.

        Default handles the standard 2-DOF state ``[R, Rdot]``.
        Override for models with different state definitions (e.g.
        ``SphericalConfinement`` with 4-DOF).
        """
        scale_factors = jnp.array([units.L_scale, units.vel_scale])
        return state * scale_factors

    # -- abstract method ---------------------------------------------------

    @abc.abstractmethod
    def __call__(
        self,
        t: Any,
        state: State,
        p_ac_fn: Callable,
    ) -> State:
        """Compute the ODE right-hand side  d(state)/dt.

        Parameters
        ----------
        t : scalar
            Current time.
        state : Array, shape (n,)
            State vector, typically ``[R, Rdot]``.
        p_ac_fn : callable  (t -> scalar)
            Driving acoustic pressure as a function of time.

        Returns
        -------
        Array, shape (n,)
            Time derivative of the state vector.
        """
        ...
