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
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

# Type alias for the ODE state vector (typically [R, Rdot]).
State = jax.Array


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------


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
        ...


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
