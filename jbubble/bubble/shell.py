"""Shell / coating models.

Computes the total inward stress from the bubble shell, including
Laplace pressure from surface tension, viscous dissipation, and
elastic restoring forces.
"""

import abc

import equinox as eqx
import jax

from .properties import ConstantProperty, Property
from .state import BubbleState


class ShellModel(eqx.Module, abc.ABC):
    """Bubble shell / coating model.

    Computes the total inward stress from the shell, including:

    - Laplace pressure from surface tension:  2 sigma(R) / R
    - Shell viscous dissipation  (e.g. 4 kappa_s Rdot / R^2)
    - Shell elastic restoring forces  (for thick shells)

    Every ``ShellModel`` holds a ``Property`` as its ``sigma`` field,
    mapping the bubble state to the effective surface tension.
    """

    sigma: Property

    def p_laplace(self, state: BubbleState) -> jax.Array:
        """Laplace pressure contribution from surface tension."""
        return 2.0 * self.sigma(state) / state.R

    @abc.abstractmethod
    def p_elastic(self, state: BubbleState) -> jax.Array:
        """Elastic contribution from the shell."""
        ...

    @abc.abstractmethod
    def p_viscous(self, state: BubbleState) -> jax.Array:
        """Viscous contribution from the shell."""
        ...

    def __call__(self, state: BubbleState) -> jax.Array:
        """Compute total shell pressure p_shell(state).

        This includes the Laplace pressure 2 sigma(R) / R plus any
        additional viscous or elastic shell terms.

        Parameters
        ----------
        state : BubbleState
            Current bubble state.

        Returns
        -------
        scalar
            Total inward shell pressure.
        """
        return self.p_laplace(state) + self.p_elastic(state) + self.p_viscous(state)


class NoShell(ShellModel):
    """No shell coating — only Laplace pressure.

    p_shell = 2 sigma(R) / R

    Suitable for uncoated gas bubbles.  Pair with a
    ``ConstantProperty`` sigma for a standard Rayleigh-Plesset setup.
    """

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0


class LipidShell(ShellModel):
    """Thin lipid shell with surface viscosity.

    p_shell = 2 sigma(R) / R  +  4 kappa_s Rdot / R^2

    This is the shell model used by Marmottant (2005) and most
    Gompertz-smoothed variants.

    Fields
    ------
    sigma : Property
        Surface tension law.
    kappa_s : Property
        Shell surface-dilatational viscosity  [N s/m].
    """

    kappa_s: Property

    @classmethod
    def marmottant_2005(
        cls,
        *,
        sigma: Property,
        kappa_s: float = 2.4e-9,
    ) -> "LipidShell":
        """Construct with literature values from Marmottant et al. (2005).

        Parameters
        ----------
        sigma : Property
            Surface tension law.
        kappa_s : float
            Shell surface viscosity  [N s/m]; default 2.4e-9.
        """
        return cls(
            sigma=sigma,
            kappa_s=ConstantProperty(val=kappa_s, _scale="kappa_scale"),
        )

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.kappa_s(state) * state.R_dot / state.R**2


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
    sigma : Property
        Surface tension law.
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
    d_s: float
    G_s: float
    mu_s: float

    @classmethod
    def church_1995(
        cls,
        *,
        sigma: Property,
        R0: float,
        d_s: float = 4e-9,
        G_s: float = 10e6,
        mu_s: float = 0.5,
    ) -> "ThickShell":
        """Construct with literature values from Church (1995).

        Parameters
        ----------
        sigma : Property
            Surface tension law.
        R0 : float
            Equilibrium bubble radius  [m].
        d_s : float
            Shell thickness  [m]; default 4e-9.
        G_s : float
            Shell shear modulus  [Pa]; default 10 MPa.
        mu_s : float
            Shell viscosity  [Pa s]; default 0.5.
        """
        return cls(sigma=sigma, R0=R0, d_s=d_s, G_s=G_s, mu_s=mu_s)

    def p_elastic(self, state: BubbleState) -> jax.Array:
        R = state.R
        return (
            (4.0 / 3.0) * self.G_s * (self.d_s / self.R0) * (1.0 - (self.R0 / R) ** 3)
        )

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu_s * self.d_s * state.R_dot / state.R**2
