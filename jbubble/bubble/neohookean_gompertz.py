"""
Neo-Hookean finite-strain viscoelastic model with Gompertz surface tension.
"""

import jax

from .kelvin_voigt_gompertz import KelvinVoigtGompertz


class NeoHookeanGompertz(KelvinVoigtGompertz):
    """
    Lipid-coated microbubble in a Neo-Hookean finite-strain medium with
    Gompertz shell surface tension.

    Identical to ``KelvinVoigtGompertz`` except the elastic medium stress
    uses the Neo-Hookean constitutive law rather than the linear
    Kelvin-Voigt formulation::

        P_elastic = (4/3) G [(R₀/R)³ − (R/R₀)³]

    This correctly captures finite-strain behaviour (large oscillation
    amplitudes) where the Kelvin-Voigt linear approximation breaks down.

    Includes:
        - Medium viscosity (μ_m) and finite-strain elasticity (G)
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law

    No liquid compressibility correction is included.
    """

    def elastic_pressure(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        """Neo-Hookean finite-strain elastic medium pressure."""
        return (4.0 / 3.0) * self.G * ((self.R0 / R) ** 3 - (R / self.R0) ** 3)
