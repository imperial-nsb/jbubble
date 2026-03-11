import jax

from . import _defaults
from .interfaces import MediumModel


class NewtonianMedium(MediumModel):
    """Newtonian liquid medium.

    p_medium = 4 mu Rdot / R

    Fields
    ------
    mu : float
        Dynamic viscosity  [Pa s].
    """

    mu: float = _defaults.MU_WATER

    def p_viscous(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return 4.0 * self.mu * R_dot / R

    def p_elastic(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return R * 0.0


class KelvinVoigtMedium(MediumModel):
    """Kelvin-Voigt viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R / R0)^3 - 1)

    Fields
    ------
    R0 : float
        Equilibrium bubble radius  [m].
    G : float
        Shear modulus  [Pa].
    mu : float
        Dynamic viscosity  [Pa s].
    """

    R0: float
    G: float
    mu: float = _defaults.MU_WATER

    def p_viscous(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return 4.0 * self.mu * R_dot / R

    def p_elastic(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return (4.0 / 3.0) * self.G * ((R / self.R0) ** 3 - 1.0)


class NeoHookeanMedium(MediumModel):
    """Neo-Hookean finite-strain viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R0/R)^3 - (R/R0)^3)

    Correctly captures finite-strain behaviour at large oscillation
    amplitudes where the Kelvin-Voigt linear approximation breaks down.

    Fields
    ------
    R0 : float
        Equilibrium bubble radius  [m].
    G : float
        Shear modulus  [Pa].
    mu : float
        Dynamic viscosity  [Pa s].
    """

    R0: float
    G: float
    mu: float = _defaults.MU_WATER

    def p_viscous(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return 4.0 * self.mu * R_dot / R

    def p_elastic(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return (4.0 / 3.0) * self.G * ((self.R0 / R) ** 3 - (R / self.R0) ** 3)
