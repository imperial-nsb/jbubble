"""Shell and coating models.

Each model computes the total inward stress from the bubble shell,
including Laplace pressure from surface tension, viscous dissipation, and
elastic restoring forces. The module also defines the state-dependent
surface tension laws
[`MarmottantSurfaceTension`][jbubble.bubble.shell.MarmottantSurfaceTension]
and [`GompertzSurfaceTension`][jbubble.bubble.shell.GompertzSurfaceTension].
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .property import Property, as_property
from .state import BubbleState

__all__ = [
    "ShellModel",
    "NoShell",
    "LipidShell",
    "ThickShell",
    "MarmottantSurfaceTension",
    "GompertzSurfaceTension",
]


class ShellModel(eqx.Module, abc.ABC):
    r"""Bubble shell or coating model.

    Computes the total inward stress from the shell, including:

    - Laplace pressure from surface tension, $2\sigma(R)/R$.
    - Shell viscous dissipation, for example $4\kappa_s\dot{R}/R^2$.
    - Shell elastic restoring forces, for thick shells.

    Every `ShellModel` holds a [`Property`][jbubble.bubble.property.Property]
    as its `sigma` field. The field's
    [`as_property`][jbubble.bubble.property.as_property] converter turns a
    plain float into a `Property`.

    Parameters
    ----------
    sigma : float or Property
        Surface tension law [N/m].
    """

    sigma: Property = eqx.field(converter=as_property)

    def p_laplace(self, state: BubbleState) -> jax.Array:
        r"""Return the Laplace pressure from surface tension, $2\sigma(R)/R$ [Pa]."""
        return 2.0 * self.sigma(state) / state.R

    @abc.abstractmethod
    def p_elastic(self, state: BubbleState) -> jax.Array:
        """Return the elastic contribution from the shell [Pa]."""
        ...

    @abc.abstractmethod
    def p_viscous(self, state: BubbleState) -> jax.Array:
        """Return the viscous contribution from the shell [Pa]."""
        ...

    def __call__(self, state: BubbleState) -> jax.Array:
        r"""Compute the total shell pressure, $p_\text{shell}(\text{state})$.

        $$
        p_\text{shell} = p_\text{Laplace} + p_\text{elastic} + p_\text{viscous}
        $$

        Parameters
        ----------
        state : BubbleState
            Current bubble state.

        Returns
        -------
        jax.Array
            Scalar total inward shell pressure [Pa].
        """
        return self.p_laplace(state) + self.p_elastic(state) + self.p_viscous(state)


class NoShell(ShellModel):
    r"""No shell coating, only Laplace pressure.

    $$
    p_\text{shell} = \frac{2\sigma(R)}{R}
    $$

    Suitable for uncoated gas bubbles. Accepts a plain float for `sigma`,
    for example `72e-3` for water.

    Parameters
    ----------
    sigma : float or Property
        Surface tension law [N/m].
    """

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0


class LipidShell(ShellModel):
    r"""Thin lipid shell with surface viscosity.

    $$
    p_\text{shell} = \frac{2\sigma(R)}{R} + \frac{4\kappa_s\dot{R}}{R^2}
    $$

    This is the shell model that Marmottant (2005) and most
    Gompertz-smoothed variants use.

    Parameters
    ----------
    sigma : float or Property
        Surface tension law [N/m].
    kappa_s : float or Property
        Shell surface-dilatational viscosity [N s/m].

    Notes
    -----
    `p_elastic` returns zero for this model. The shell elasticity isn't
    absent: the surface tension law `sigma` encodes all of it. When `sigma`
    is state-dependent, as with
    [`MarmottantSurfaceTension`][jbubble.bubble.shell.MarmottantSurfaceTension]
    or [`GompertzSurfaceTension`][jbubble.bubble.shell.GompertzSurfaceTension],
    the area-elasticity term $\chi\left[(R/R_b)^2 - 1\right]$ enters
    through the Laplace pressure $2\sigma(R)/R$, not through a separate
    `p_elastic` term. This matches how the literature writes the
    Marmottant model: the elastic and ruptured regimes modify $\sigma(R)$
    rather than adding an independent stress contribution.
    """

    kappa_s: Property = eqx.field(converter=as_property)

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.kappa_s(state) * state.R_dot / state.R**2


class ThickShell(ShellModel):
    r"""Church (1995) thick viscoelastic shell.

    In addition to Laplace pressure, this model includes thick-shell
    elastic and viscous contributions:

    $$
    \begin{aligned}
    p_\text{elastic} &= \frac{4}{3} G_s \frac{d_s}{R_0}
        \left[1 - \left(\frac{R_0}{R}\right)^3\right], \\
    p_\text{viscous} &= \frac{4\mu_s d_s \dot{R}}{R^2}.
    \end{aligned}
    $$

    The total shell pressure is

    $$
    p_\text{shell} = \frac{2\sigma(R)}{R} + p_\text{elastic} + p_\text{viscous}.
    $$

    Parameters
    ----------
    sigma : float or Property
        Surface tension law [N/m].
    d_s : float or Property
        Shell thickness [m].
    G_s : float or Property
        Shell shear modulus [Pa]. It can be state-dependent, for example to
        model strain stiffening or strain softening.
    mu_s : float or Property
        Shell viscosity [Pa s]. It can be state-dependent, for example to
        model shear thinning.
    """

    d_s: Property = eqx.field(converter=as_property)
    G_s: Property = eqx.field(converter=as_property)
    mu_s: Property = eqx.field(converter=as_property)

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


class MarmottantSurfaceTension(Property):
    r"""Piecewise Marmottant surface tension law.

    Three regimes, based on the radius $R$ relative to the buckling radius
    $R_b$ and the rupture radius $R_r$:

    $$
    \sigma(R) =
    \begin{cases}
    0 & R \le R_b \quad \text{(buckled)}, \\
    \chi\left[(R/R_b)^2 - 1\right] & R_b < R < R_r \quad \text{(elastic)}, \\
    \sigma_r & R \ge R_r \quad \text{(ruptured)},
    \end{cases}
    $$

    where $R_b$ is `R_buckle_ratio * state.R0`, $\sigma_r$ is
    `sigma_rupture`, and continuity of $\sigma$ at the elastic-to-ruptured
    transition gives $R_r = R_b\sqrt{1 + \sigma_r/\chi}$.

    Parameters
    ----------
    R_buckle_ratio : float
        Buckling radius as a fraction of `R0` (dimensionless).
    chi : float
        Shell elasticity [N/m].
    sigma_rupture : float
        Surface tension after rupture [N/m].

    Notes
    -----
    $\sigma(R)$ has discontinuous first derivatives at the regime
    boundaries. For applications that need smooth gradients, such as
    gradient-based optimisation, use
    [`GompertzSurfaceTension`][jbubble.bubble.shell.GompertzSurfaceTension]
    instead.
    """

    R_buckle_ratio: float
    chi: float
    sigma_rupture: float

    def __call__(self, state: BubbleState) -> jax.Array:
        R, R0 = state.R, state.R0
        R_buckle = self.R_buckle_ratio * R0
        chi = self.chi
        sigma_rupture = self.sigma_rupture
        R_rupture = R_buckle * jnp.sqrt(sigma_rupture / chi + 1.0)
        sigma_elastic = chi * ((R / R_buckle) ** 2 - 1.0)
        in_elastic = (R_buckle < R) & (R_rupture > R)
        in_ruptured = R_rupture <= R
        return jnp.where(
            in_ruptured,
            sigma_rupture,
            jnp.where(
                in_elastic,
                sigma_elastic,
                0.0,
            ),
        )


class GompertzSurfaceTension(Property):
    r"""Smooth Gompertz surface tension law.

    A later release redesigns this class, so its parameters and defaults
    may change.

    A differentiable Gompertz function approximates the piecewise
    Marmottant surface tension, which keeps automatic differentiation
    robust:

    $$
    \sigma(R) = a \exp\left[-b \exp\left(c\left(1 - \frac{R}{R_b}\right)\right)\right]
    $$

    where $R_b$ is `R_buckle_ratio * state.R0`. The code derives the
    Gompertz parameters from $\chi$ (`chi`), $\sigma_r$ (`sigma_rupture`),
    and the dimensionless `sharpness` factor $s$:

    $$
    \begin{aligned}
    a &= \sigma_r, \\
    c &= s \, \frac{2\chi}{\sigma_r} \sqrt{1 + \frac{\sigma_r}{2\chi}}, \\
    b &= -\ln\left(\frac{\sigma_0}{\sigma_r}\right)
        \exp\left[-c\left(1 - \frac{R_0}{R_b}\right)\right],
    \qquad
    \sigma_0 = \chi\left[\left(\frac{R_0}{R_b}\right)^2 - 1\right].
    \end{aligned}
    $$

    So $\sigma(R_0) = \sigma_0$ matches the Marmottant elastic regime, and
    $\sigma \to \sigma_r$ as $R \to \infty$. The model reads $R_0$ from
    the state, so it stays consistent when $R_0$ evolves, for example
    through rectified diffusion.

    The `sharpness` factor sets the steepness of the transition through
    $c$. Scaling $c$ leaves both anchors fixed: the code re-solves $b$ so
    that $\sigma(R_0)$ stays the same, and $\sigma \to \sigma_r$ as
    $R \to \infty$ regardless.

    Parameters
    ----------
    R_buckle_ratio : float
        Buckling radius as a fraction of `R0` (dimensionless).
    chi : float
        Shell elasticity [N/m].
    sigma_rupture : float
        Asymptotic (ruptured) surface tension [N/m].
    sharpness : float
        Dimensionless multiplier on the transition rate $c$.
        Default: `1.0`.

    Raises
    ------
    ValueError
        If $\sigma_0 \ge \sigma_r$ at construction, that is, the bubble
        starts in the ruptured regime. If you construct the model inside
        `jax.jit`, JAX raises it as `jax.errors.JaxRuntimeError`, a
        `RuntimeError` subclass.

    Notes
    -----
    The Gompertz fit requires the initial surface tension at $R_0$ to lie
    strictly below the rupture threshold:

    $$
    \chi\left[\left(\frac{1}{r_b}\right)^2 - 1\right] < \sigma_r,
    \qquad r_b = \mathtt{R\_buckle\_ratio}.
    $$

    A common mistake is to set `R_buckle_ratio` too small, for example
    `0.9`, which inflates $\sigma(R_0)$ above `sigma_rupture`. Values
    around 0.95 to 0.99 are typical.
    """

    R_buckle_ratio: float
    chi: float
    sigma_rupture: float
    sharpness: float = 1.0

    def __post_init__(self) -> None:
        sigma_at_R0 = self.chi * ((1.0 / self.R_buckle_ratio) ** 2 - 1.0)

        def _check(s_at_r0, s_rupture):
            if s_at_r0 >= s_rupture:
                raise ValueError(
                    f"GompertzSurfaceTension: sigma(R0) = {s_at_r0:.4g} N/m "
                    f">= sigma_rupture = {s_rupture:.4g} N/m.  "
                    f"The bubble starts in the ruptured regime and the Gompertz "
                    f"fit is ill-posed.  Increase R_buckle_ratio (try 0.98) or "
                    f"decrease chi."
                )

        jax.debug.callback(_check, sigma_at_R0, self.sigma_rupture)

    def __call__(self, state: BubbleState) -> jax.Array:
        R, R0 = state.R, state.R0
        R_buckle = self.R_buckle_ratio * R0
        chi = self.chi
        a = self.sigma_rupture
        c = self.sharpness * (2.0 * chi / a) * jnp.sqrt(1.0 + a / (2.0 * chi))
        sigma_R0 = chi * ((R0 / R_buckle) ** 2 - 1.0)
        b = -jnp.log(sigma_R0 / a) / jnp.exp(c * (1.0 - R0 / R_buckle))
        return a * jnp.exp(-b * jnp.exp(c * (1.0 - R / R_buckle)))
