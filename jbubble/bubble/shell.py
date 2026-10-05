"""Shell and coating models.

Each model computes the total inward stress from the bubble shell,
including Laplace pressure from surface tension, viscous dissipation, and
elastic restoring forces. The module also defines the state-dependent
surface tension laws for lipid shells:

- [`MarmottantSurfaceTension`][jbubble.bubble.shell.MarmottantSurfaceTension]:
  the piecewise Marmottant law, with buckled, elastic, and ruptured
  regimes.
- [`SmoothMarmottantSurfaceTension`][jbubble.bubble.shell.SmoothMarmottantSurfaceTension]:
  the Marmottant law with smoothed corners, for gradient-based fitting.
- [`GompertzSurfaceTension`][jbubble.bubble.shell.GompertzSurfaceTension]:
  the Marmottant-Gompertz law of Gümmer et al. (2021).
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .property import ConstantProperty, Property, as_property
from .state import BubbleState

__all__ = [
    "ShellModel",
    "NoShell",
    "LipidShell",
    "ThickShell",
    "MarmottantSurfaceTension",
    "SmoothMarmottantSurfaceTension",
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

    This is the shell model of Marmottant et al. (2005) and of its smoothed
    variants.

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
    or
    [`SmoothMarmottantSurfaceTension`][jbubble.bubble.shell.SmoothMarmottantSurfaceTension],
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

    The surface tension at $R_0$ is
    $\sigma_0 = \chi\left[(1/r_b)^2 - 1\right]$, where $r_b$ is
    `R_buckle_ratio`, so $r_b = (1 + \sigma_0/\chi)^{-1/2}$. A ratio of 1
    or more starts the bubble buckled.

    Parameters
    ----------
    R_buckle_ratio : float or jax.Array
        Buckling radius as a fraction of `R0` (dimensionless).
    chi : float or Property
        Shell elasticity [N/m].
    sigma_rupture : float or Property
        Surface tension after rupture [N/m].

    Notes
    -----
    $\sigma(R)$ has discontinuous first derivatives at the regime
    boundaries. For applications that need smooth gradients, such as
    gradient-based optimisation, use
    [`SmoothMarmottantSurfaceTension`][jbubble.bubble.shell.SmoothMarmottantSurfaceTension]
    instead.

    The [`as_property`][jbubble.bubble.property.as_property] converter
    stores `chi` and `sigma_rupture` as
    [`Property`][jbubble.bubble.property.Property] instances. To replace
    one in an existing model with `eqx.tree_at`, which bypasses the
    converter, pass a `Property` or target its `val` leaf.

    References
    ----------
    Marmottant, P., van der Meer, S., Emmer, M., Versluis, M., de Jong, N.,
    Hilgenfeldt, S., & Lohse, D. (2005). A model for large amplitude
    oscillations of coated bubbles accounting for buckling and rupture.
    *J. Acoust. Soc. Am.* 118(6), 3499-3505.
    [doi:10.1121/1.2109427](https://doi.org/10.1121/1.2109427)
    """

    R_buckle_ratio: ArrayLike
    chi: Property = eqx.field(converter=as_property)
    sigma_rupture: Property = eqx.field(converter=as_property)

    def __call__(self, state: BubbleState) -> jax.Array:
        R, R0 = state.R, state.R0
        R_buckle = self.R_buckle_ratio * R0
        chi = self.chi(state)
        sigma_rupture = self.sigma_rupture(state)
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


class SmoothMarmottantSurfaceTension(Property):
    r"""Marmottant surface tension law with smoothed corners.

    The piecewise Marmottant law clips the elastic surface tension to the
    interval $[0, \sigma_r]$. This law, which is jbubble's own
    construction, replaces the hard clip with a logistic-smoothed clamp
    whose width $\varepsilon$, set by `smoothing`, is a fraction of
    $\sigma_r$:

    $$
    \begin{aligned}
    y(R) &= \frac{\chi}{\sigma_r}\left[\left(\frac{R}{R_b}\right)^2 - 1\right],
    \qquad R_b = r_b R_0, \\
    \sigma(R) &= \sigma_r \varepsilon\left[
        \operatorname{softplus}\left(\frac{y}{\varepsilon}\right)
        - \operatorname{softplus}\left(\frac{y - 1}{\varepsilon}\right)\right],
    \end{aligned}
    $$

    where $r_b$ is `R_buckle_ratio`, $\chi$ is `chi`, $\sigma_r$ is
    `sigma_rupture`, and $\operatorname{softplus}(x) = \ln(1 + e^x)$.
    Equivalently, $\sigma$ is the Marmottant law averaged over a logistic
    spread of scale $\varepsilon\sigma_r$ in the elastic surface tension,
    which rounds the buckling and rupture corners.

    The law has the following properties:

    - It's infinitely differentiable in $R$, $\chi$, $r_b$, and
      $\sigma_r$.
    - It's non-decreasing in $R$ and stays within $[0, \sigma_r]$.
    - It differs from the Marmottant law by at most
      $\varepsilon \ln 2 \, \sigma_r$ for every $\chi$ and $r_b$, and it
      reaches that bound at the two corners. With the default
      `smoothing = 0.01` and $\sigma_r = 0.072$ N/m, the bound is
      0.5 mN/m.
    - It converges uniformly to the Marmottant law as
      $\varepsilon \to 0$.
    - It's well-posed for every parameter value, including bubbles that
      start buckled (`R_buckle_ratio` of 1 or more) or ruptured.

    Use this law for gradient-based fitting of a lipid shell. Use
    [`MarmottantSurfaceTension`][jbubble.bubble.shell.MarmottantSurfaceTension]
    for the exact piecewise law in forward simulations.

    Parameters
    ----------
    R_buckle_ratio : float or jax.Array
        Buckling radius as a fraction of `R0` (dimensionless).
    chi : float or Property
        Shell elasticity [N/m].
    sigma_rupture : float or Property
        Surface tension after rupture [N/m].
    smoothing : float or jax.Array
        Corner width $\varepsilon$ as a fraction of `sigma_rupture`
        (dimensionless). Must be positive. Default: `0.01`.

    Raises
    ------
    ValueError
        If `smoothing` isn't positive. The constructor checks concrete
        values only, and skips the check when `smoothing` is a JAX tracer.

    Notes
    -----
    The default `smoothing = 0.01` keeps the law close to the piecewise
    one without stiffening the ODE. Keller-Miksis benchmarks with
    `R_buckle_ratio = 0.98`, `R0` from 1 to 3 µm, 1 to 3 MHz, and 25 kPa
    to 1 MPa show that, at the default:

    - The radius curve differs from the piecewise law's by about as much
      as the piecewise law's own error at the default solver tolerances.
    - The derivative of the peak radius with respect to $\chi$ has the
      same sign as the piecewise law's in every case tested.
    - Fitting $\chi$ to radius curves from the piecewise law at 50 kPa
      recovers it to within 0.1%.
    - The solver takes no more steps than it does for the piecewise law.

    The smoothing biases the result when the bubble starts near a corner,
    that is, when $\sigma_0 = \sigma(R_0)$ of the piecewise law lies within
    about $5\varepsilon\sigma_r$ of 0 or of $\sigma_r$. For example,
    `R_buckle_ratio` close to 1 starts the bubble near buckling. Then
    $\sigma(R_0)$ shifts by up to $\varepsilon \ln 2 \, \sigma_r$, and the
    elasticity at $R_0$ drops to as little as half the Marmottant value.
    At `smoothing = 0.01`, this gives a radius bias of about $10^{-3} R_0$
    (root mean square) and a bias in fitted $\chi$ of a few percent, up to
    8% at 200 kPa. In that regime, use `smoothing = 0.005`, or use
    [`MarmottantSurfaceTension`][jbubble.bubble.shell.MarmottantSurfaceTension]
    for forward-only simulations. The bubble still starts at equilibrium,
    because
    [`initial_state`][jbubble.bubble.eom.EquationOfMotion.initial_state]
    reads $\sigma(R_0)$ from the law itself.

    The [`as_property`][jbubble.bubble.property.as_property] converter
    stores `chi` and `sigma_rupture` as
    [`Property`][jbubble.bubble.property.Property] instances.
    """

    R_buckle_ratio: ArrayLike
    chi: Property = eqx.field(converter=as_property)
    sigma_rupture: Property = eqx.field(converter=as_property)
    smoothing: ArrayLike = 0.01

    def __check_init__(self) -> None:
        smoothing = _concrete_value(self.smoothing)
        if smoothing is not None and not np.all(smoothing > 0.0):
            raise ValueError(
                "SmoothMarmottantSurfaceTension needs smoothing > 0, got "
                f"{_format_values(smoothing)}."
            )

    def __call__(self, state: BubbleState) -> jax.Array:
        R_b = self.R_buckle_ratio * state.R0
        sigma_r = self.sigma_rupture(state)
        y = self.chi(state) * ((state.R / R_b) ** 2 - 1.0) / sigma_r
        return sigma_r * _smooth_clamp_unit(y, self.smoothing)


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


def _concrete_value(x: object) -> np.ndarray | None:
    """Return `x` as a NumPy array, or `None` if its value isn't known yet.

    Construction-time checks use this helper to validate plain Python,
    NumPy, and concrete JAX values without adding anything to a JAX trace.
    It returns `None` for a JAX tracer, and for a
    [`Property`][jbubble.bubble.property.Property] other than a
    [`ConstantProperty`][jbubble.bubble.property.ConstantProperty], whose
    value depends on the bubble state.
    """
    if isinstance(x, ConstantProperty):
        x = x.val
    elif isinstance(x, Property):
        return None
    try:
        return np.asarray(x, dtype=np.float64)
    except jax.errors.TracerArrayConversionError:
        return None


def _format_values(x: np.ndarray) -> str:
    """Format a scalar or an array of parameter values for an error message."""
    if x.ndim == 0:
        return f"{float(x):.4g}"
    return np.array2string(x, precision=4)


def _smooth_clamp_unit(y: jax.Array, eps: ArrayLike) -> jax.Array:
    r"""Return a logistic-smoothed clamp of `y` to the unit interval.

    $$
    f(y) = \varepsilon\left[\operatorname{softplus}(y/\varepsilon)
        - \operatorname{softplus}((y - 1)/\varepsilon)\right]
    $$

    $f(y)$ is the expected value of
    $\operatorname{clip}(y + \varepsilon L, 0, 1)$ for a standard logistic
    random variable $L$. It's smooth and strictly increasing, it satisfies
    $f(y) + f(1 - y) = 1$, and it differs from $\operatorname{clip}(y, 0, 1)$
    by at most $\varepsilon \ln 2$.

    The function evaluates the upper half through that symmetry, so the
    two softplus terms never cancel catastrophically. It picks the half
    with `jnp.where` rather than `jnp.minimum(y, 1 - y)`, because
    `jnp.minimum` splits the gradient at the tie $y = 1/2$ and gives a zero
    derivative there.
    """
    lower_half = y <= 0.5
    lo = jnp.where(lower_half, y, 1.0 - y)
    h = eps * (jax.nn.softplus(lo / eps) - jax.nn.softplus((lo - 1.0) / eps))
    return jnp.where(lower_half, h, 1.0 - h)
