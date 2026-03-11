import jax

from . import _defaults
from .interfaces import ShellModel


class NoShell(ShellModel):
    """No shell coating — only Laplace pressure.

    p_shell = 2 sigma(R) / R

    Suitable for uncoated gas bubbles.  Pair with ``ConstantSigma``
    for a standard Rayleigh-Plesset setup, or with any other
    ``SurfaceTensionModel`` if desired.
    """

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return 2.0 * self.sigma(R) / R


class LipidShell(ShellModel):
    """Thin lipid shell with surface viscosity.

    p_shell = 2 sigma(R) / R  +  4 kappa_s Rdot / R^2

    This is the shell model used by Marmottant (2005) and most
    Gompertz-smoothed variants.  Pair with ``MarmottantSigma`` for
    the original piecewise law or ``GompertzSigma`` for smooth autodiff.

    Fields
    ------
    sigma : SurfaceTensionModel
        Surface tension law (inherited from ShellModel).
    kappa_s : float
        Shell surface-dilatational viscosity  [N s/m].
    """

    kappa_s: float = _defaults.KAPPA_S_LIPID

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        return 2.0 * self.sigma(R) / R + 4.0 * self.kappa_s * R_dot / R**2


class ThickShell(ShellModel):
    """Church (1995) thick viscoelastic shell.

    In addition to Laplace pressure, this model includes thick-shell
    elastic and viscous contributions::

        p_elastic    = (4/3) G_s (d_s / R0) (1 - (R0/R)^3)
        p_shell_visc = 4 mu_s d_s Rdot / R^2

    Total shell pressure::

        p_shell = 2 sigma(R) / R  +  p_elastic  +  p_shell_visc

    Fields
    ------
    sigma : SurfaceTensionModel
        Surface tension law (inherited from ShellModel).
    R0 : float
        Equilibrium bubble radius  [m].
    d_s : float
        Shell thickness  [m].
    G_s : float
        Shell shear modulus  [Pa].
    mu_s : float
        Shell viscosity  [Pa s].
    """

    R0: float
    d_s: float = 4e-9
    G_s: float = 10e6
    mu_s: float = 0.5

    def __call__(self, R: jax.Array, R_dot: jax.Array) -> jax.Array:
        p_laplace = 2.0 * self.sigma(R) / R
        p_elastic = (
            (4.0 / 3.0) * self.G_s * (self.d_s / self.R0) * (1.0 - (self.R0 / R) ** 3)
        )
        p_visc = 4.0 * self.mu_s * self.d_s * R_dot / R**2
        return p_laplace + p_elastic + p_visc
