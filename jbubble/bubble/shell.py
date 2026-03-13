"""Shell / coating models.

Computes the total inward stress from the bubble shell, including
Laplace pressure from surface tension, viscous dissipation, and
elastic restoring forces.
"""

import abc
import dataclasses

import equinox as eqx
import jax

from .properties import Property, as_property
from .state import BubbleState


class ShellModel(eqx.Module, abc.ABC):
    """Bubble shell / coating model.

    Computes the total inward stress from the shell, including:

    - Laplace pressure from surface tension:  2 sigma(R) / R
    - Shell viscous dissipation  (e.g. 4 kappa_s Rdot / R^2)
    - Shell elastic restoring forces  (for thick shells)

    Every ``ShellModel`` holds a ``Property`` as its ``sigma`` field.
    A plain float is accepted and auto-converted to a ``ConstantProperty``
    in ``__post_init__``.

    Subclasses can declare additional ``Property`` fields by annotating
    them with ``eqx.field(metadata={"property_scale": "<scale_name>"})``
    — the base ``__post_init__`` will convert them automatically.
    No subclass ``__post_init__`` is needed.
    """

    sigma: Property = eqx.field(metadata={"property_scale": "sigma_scale"})

    def __post_init__(self):
        for f in dataclasses.fields(self):
            scale = f.metadata.get("property_scale")
            if scale is not None:
                object.__setattr__(
                    self, f.name, as_property(getattr(self, f.name), scale)
                )

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

    Suitable for uncoated gas bubbles.  Accepts a plain float for
    ``sigma`` (e.g. 72e-3 for water).

    Fields
    ------
    sigma : float or Property
        Surface tension law.
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
    sigma : float or Property
        Surface tension law.
    kappa_s : float or Property
        Shell surface-dilatational viscosity  [N s/m].
    """

    kappa_s: Property = eqx.field(metadata={"property_scale": "kappa_scale"})

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
    sigma : float or Property
        Surface tension law.
    d_s : float or Property
        Shell thickness  [m].
    G_s : float or Property
        Shell shear modulus  [Pa].  May be state-dependent (e.g. strain-
        stiffening / strain-softening).
    mu_s : float or Property
        Shell viscosity  [Pa s].  May be state-dependent (e.g. shear-
        thinning).
    """

    d_s: Property = eqx.field(metadata={"property_scale": "length_scale"})
    G_s: Property = eqx.field(metadata={"property_scale": "P_scale"})
    mu_s: Property = eqx.field(metadata={"property_scale": "mu_scale"})

    def p_elastic(self, state: BubbleState) -> jax.Array:
        R = state.R
        return (
            (4.0 / 3.0)
            * self.G_s(state)
            * (self.d_s(state) / state.R0)
            * (1.0 - (state.R0 / R) ** 3)
        )

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu_s(state) * self.d_s(state) * state.R_dot / state.R**2
