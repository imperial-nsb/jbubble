import jax

from . import _defaults
from .interfaces import MediumModel


class KelvinVoigtMedium(MediumModel):
    """Kelvin-Voigt viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R / R0)^3 - 1)

    When G = 0 (default), reduces to a Newtonian liquid:  4 mu Rdot / R.

    Fields
    ------
    R0 : float
        Equilibrium bubble radius  [m].
    mu : float
        Dynamic viscosity  [Pa s].
    G : float
        Shear modulus  [Pa].  Zero for a Newtonian liquid.
    """

    R0: float
    mu: float = _defaults.MU_WATER
    G: float = 0.0

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        p_viscous = 4.0 * self.mu * R_dot / R
        p_elastic = (4.0 / 3.0) * self.G * ((R / self.R0) ** 3 - 1.0)
        return p_viscous + p_elastic


class NeoHookeanMedium(MediumModel):
    """Neo-Hookean finite-strain viscoelastic medium.

    p_medium = 4 mu Rdot / R  +  (4 G / 3) ((R0/R)^3 - (R/R0)^3)

    Correctly captures finite-strain behaviour at large oscillation
    amplitudes where the Kelvin-Voigt linear approximation breaks down.
    When G = 0 (default), reduces to a Newtonian liquid.

    Fields
    ------
    R0 : float
        Equilibrium bubble radius  [m].
    mu : float
        Dynamic viscosity  [Pa s].
    G : float
        Shear modulus  [Pa].  Zero for a Newtonian liquid.
    """

    R0: float
    mu: float = _defaults.MU_WATER
    G: float = 0.0

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        p_viscous = 4.0 * self.mu * R_dot / R
        p_elastic = (4.0 / 3.0) * self.G * ((self.R0 / R) ** 3 - (R / self.R0) ** 3)
        return p_viscous + p_elastic
