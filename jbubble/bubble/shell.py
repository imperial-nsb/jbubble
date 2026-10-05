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
    r"""Incompressible viscoelastic shell of finite thickness (Church 1995).

    The shell is a layer of incompressible Kelvin-Voigt solid with shear
    modulus $G_s$, shear viscosity $\mu_s$, and thickness $d_s$ at the
    equilibrium radius $R_0$. Integrating its radial stress across the
    layer gives the shell stresses of the Church (1995) model in the
    finite-deformation form of Qin and Ferrara (2010, Eq. 10):

    $$
    \begin{aligned}
    p_\text{elastic} &= \frac{4}{3} G_s
        \left[1 - \left(\frac{R_0}{R}\right)^3\right]
        \frac{V_s}{R^3 - V_s}, \\
    p_\text{viscous} &= 4 \mu_s \frac{V_s}{R^3 - V_s} \frac{\dot{R}}{R},
    \end{aligned}
    \qquad
    V_s = R_0^3 - (R_0 - d_s)^3,
    $$

    where $R$ is the outer, liquid-side radius and $V_s$ is the shell
    volume divided by $4\pi/3$. Because the shell is incompressible, $V_s$
    stays constant, and $R^3 - V_s$ is the cube of the inner radius. The
    total shell pressure is

    $$
    p_\text{shell} = \frac{2\sigma(R)}{R} + p_\text{elastic} + p_\text{viscous}.
    $$

    For a thin shell, $d_s \ll R_0$, so $V_s \approx 3 R_0^2 d_s$, and the
    model reduces to the thin-shell model of Hoff et al. (2000):

    $$
    p_\text{elastic} \approx 12 G_s d_s \frac{R_0^2}{R^3}
        \left(1 - \frac{R_0}{R}\right),
    \qquad
    p_\text{viscous} \approx 12 \mu_s d_s \frac{R_0^2 \dot{R}}{R^4}.
    $$

    The viscous terms agree exactly in that limit, and the elastic terms
    agree to first order in the strain. For small oscillations about
    $R_0$, the shell therefore adds a stiffness of $12 G_s d_s / R_0$ per
    unit radial strain and a damping of $12 \mu_s d_s / R_0^2$ per unit
    wall velocity, with relative corrections of order $d_s/R_0$.

    Parameters
    ----------
    sigma : float or Property
        Surface tension law [N/m].
    d_s : float or Property
        Shell thickness at the equilibrium radius $R_0$ [m]. The shell
        volume, not the thickness, stays constant as the bubble moves.
    G_s : float or Property
        Shell shear modulus [Pa]. It can be state-dependent, for example to
        model strain stiffening or strain softening.
    mu_s : float or Property
        Shell shear viscosity [Pa s]. It can be state-dependent, for
        example to model shear thinning.

    Notes
    -----
    jbubble integrates a single radius, so this class simplifies the full
    Church and Qin-Ferrara equations as follows:

    - The gas model reads the outer radius $R$ rather than the inner
      radius $(R^3 - V_s)^{1/3}$, so the gas stiffness comes out low by a
      relative amount of about $3 d_s / R_0$.
    - The class neglects the shell's inertia, a relative error of order
      $d_s / R_0$ in the inertial terms.
    - A single surface tension $\sigma$ acts at $R$, in place of separate
      gas-shell and shell-liquid tensions.
    - The shell is unstrained at $R_0$, and
      [`initial_state`][jbubble.bubble.eom.EquationOfMotion.initial_state]
      balances the Laplace pressure with the gas pressure.

    The viscous term is identical to Church's
    $4 \mu_s V_s \dot{R}_1 / (R_1 R^3)$, written with the outer radius.
    The elastic term agrees with Church's
    $4 G_s (V_s / R^3)(1 - R_{1,0}/R_1)$, quoted by Tu et al. (2009,
    Eq. 2), to first order in the strain. Here $R_1$ is the inner radius
    and $R_{1,0} = R_0 - d_s$. The model needs $R^3 > V_s$, that is, a
    positive inner radius.

    References
    ----------
    Church, C. C. (1995). The effects of an elastic solid surface layer on
    the radial pulsations of gas bubbles. *J. Acoust. Soc. Am.* 97(3),
    1510-1521. [doi:10.1121/1.412091](https://doi.org/10.1121/1.412091)

    Hoff, L., Sontum, P. C., & Hovem, J. M. (2000). Oscillations of
    polymeric microbubbles: Effect of the encapsulating shell. *J. Acoust.
    Soc. Am.* 107(4), 2272-2280.
    [doi:10.1121/1.428557](https://doi.org/10.1121/1.428557)

    Qin, S., & Ferrara, K. W. (2010). A model for the dynamics of
    ultrasound contrast agents in vivo. *J. Acoust. Soc. Am.* 128(3),
    1511-1521. [doi:10.1121/1.3409476](https://doi.org/10.1121/1.3409476)

    Tu, J., Guan, J., Qiu, Y., & Matula, T. J. (2009). Estimating the shell
    parameters of SonoVue microbubbles using light scattering. *J. Acoust.
    Soc. Am.* 126(6), 2954-2962.
    [doi:10.1121/1.3242346](https://doi.org/10.1121/1.3242346)
    """

    d_s: Property = eqx.field(converter=as_property)
    G_s: Property = eqx.field(converter=as_property)
    mu_s: Property = eqx.field(converter=as_property)

    def _volume_factor(self, state: BubbleState) -> jax.Array:
        r"""Return $V_s / (R^3 - V_s)$, the shell-to-inner-gas volume ratio."""
        R0, d_s = state.R0, self.d_s(state)
        # R0^3 - (R0 - d_s)^3, factored to avoid cancellation for thin shells.
        V_s = d_s * (3.0 * R0**2 - 3.0 * R0 * d_s + d_s**2)
        return V_s / (state.R**3 - V_s)

    def p_elastic(self, state: BubbleState) -> jax.Array:
        strain = 1.0 - (state.R0 / state.R) ** 3
        return (4.0 / 3.0) * self.G_s(state) * strain * self._volume_factor(state)

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return (
            4.0 * self.mu_s(state) * self._volume_factor(state) * state.R_dot / state.R
        )


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
    - It's well-posed for all positive `R_buckle_ratio`, `chi`, and
      `sigma_rupture`, including bubbles that start buckled
      (`R_buckle_ratio` of 1 or more) or ruptured.

    Use this law for gradient-based fitting of a lipid shell. Use
    [`MarmottantSurfaceTension`][jbubble.bubble.shell.MarmottantSurfaceTension]
    for the exact piecewise law in forward simulations.

    Parameters
    ----------
    R_buckle_ratio : float or jax.Array
        Buckling radius as a fraction of `R0` (dimensionless). Must be
        positive.
    chi : float or Property
        Shell elasticity [N/m]. Must be positive.
    sigma_rupture : float or Property
        Surface tension after rupture [N/m]. Must be positive.
    smoothing : float or jax.Array
        Corner width $\varepsilon$ as a fraction of `sigma_rupture`
        (dimensionless). Must be positive. Default: `0.01`.

    Raises
    ------
    ValueError
        If `R_buckle_ratio`, `chi`, `sigma_rupture`, or `smoothing` isn't
        positive. The constructor checks concrete values only. It skips
        the check for a value that's a JAX tracer, for example inside
        `jax.jit` or `jax.grad`, or a state-dependent
        [`Property`][jbubble.bubble.property.Property], so keep such values
        in range yourself.

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
        for name in ("R_buckle_ratio", "chi", "sigma_rupture", "smoothing"):
            value = _concrete_value(getattr(self, name))
            if value is not None and not np.all(value > 0.0):
                raise ValueError(
                    f"SmoothMarmottantSurfaceTension needs {name} > 0, got "
                    f"{_format_values(value)}."
                )

    def __call__(self, state: BubbleState) -> jax.Array:
        R_b = self.R_buckle_ratio * state.R0
        sigma_r = self.sigma_rupture(state)
        y = self.chi(state) * ((state.R / R_b) ** 2 - 1.0) / sigma_r
        return sigma_r * _smooth_clamp_unit(y, self.smoothing)


class GompertzSurfaceTension(Property):
    r"""Marmottant-Gompertz surface tension law (Gümmer et al. 2021).

    A Gompertz function replaces the corners of the piecewise Marmottant
    law with a smooth curve that takes the same inputs (Gümmer, Schenke,
    and Denner 2021, Eqs. 11 and 13-15):

    $$
    \sigma(R) = \sigma_r \exp\left[-b
        \exp\left(c\left(1 - \frac{R}{R_b}\right)\right)\right],
    $$

    $$
    \begin{aligned}
    R_b &= r_b R_0,
    \qquad
    \sigma_0 = \chi\left[\left(\frac{R_0}{R_b}\right)^2 - 1\right], \\
    b &= -\ln\left(\frac{\sigma_0}{\sigma_r}\right)
        \exp\left[-c\left(1 - \frac{R_0}{R_b}\right)\right], \\
    c &= \frac{2\chi e}{\sigma_r}\sqrt{1 + \frac{\sigma_r}{2\chi}},
    \end{aligned}
    $$

    where $r_b$ is `R_buckle_ratio`, $\chi$ is `chi`, and $\sigma_r$ is
    `sigma_rupture` (the paper's $\sigma_c$). The coefficient $b$ pins
    $\sigma(R_0) = \sigma_0$, the Marmottant elastic value. The
    coefficient $c$ makes the maximum slope of the curve,
    $c\,\sigma_r/(e R_b)$ where $\sigma = \sigma_r/e$, equal to the
    Marmottant slope at $R = R_b\sqrt{1 + \sigma_r/(2\chi)}$. The paper
    takes $\sigma_0$ as its input; `R_buckle_ratio` carries the same
    information, with $r_b = (1 + \sigma_0/\chi)^{-1/2}$ (Eq. 11).

    The class evaluates the algebraically identical form

    $$
    \sigma(R) = \sigma_r \exp\left[\ln\left(\frac{\sigma_0}{\sigma_r}\right)
        \exp\left(\frac{c\,(R_0 - R)}{R_b}\right)\right],
    $$

    and caps the inner exponent at 50, so a deep compression gives
    $\sigma = 0$ with a finite gradient instead of an overflow.

    Use this law to reproduce Gümmer et al. (2021). For gradient-based
    fitting, use
    [`SmoothMarmottantSurfaceTension`][jbubble.bubble.shell.SmoothMarmottantSurfaceTension],
    which converges to the Marmottant law. The Notes explain why.

    Parameters
    ----------
    R_buckle_ratio : float or jax.Array
        Buckling radius as a fraction of `R0` (dimensionless). Must lie
        strictly between 0 and 1.
    chi : float or Property
        Shell elasticity [N/m].
    sigma_rupture : float or Property
        Asymptotic (ruptured) surface tension [N/m].

    Raises
    ------
    ValueError
        If the law is ill-posed: it needs $\chi > 0$ and
        $0 < \sigma_0 < \sigma_r$, so `R_buckle_ratio` must lie strictly
        between 0 and 1. The constructor checks concrete values only. It
        skips the check when a parameter is a JAX tracer, for example
        inside `jax.jit` or `jax.grad`, or when `chi` or `sigma_rupture` is
        a state-dependent [`Property`][jbubble.bubble.property.Property],
        so keep such values in range yourself.

    Notes
    -----
    The curve is anchored at $R_0$, not at the Marmottant transitions, so
    no value of $c$ makes it converge to the Marmottant law: as $c$ grows,
    the curve tends to a step at $R_0$. Its elasticity at $R_0$ matches
    the Marmottant value only approximately, when $\sigma_0$ is near
    $\sigma_r/e$, and it falls towards zero as $\sigma_0$ approaches 0 or
    $\sigma_r$. With `R_buckle_ratio` fixed, $\sigma_0$ grows in
    proportion to $\chi$, so the mismatch changes as you vary $\chi$.
    Fitted to radius curves from the Marmottant law, this law's $\chi$
    comes out biased by about 6% at `R_buckle_ratio = 0.98` and by up to
    about 47% at `R_buckle_ratio = 0.995`.

    The [`as_property`][jbubble.bubble.property.as_property] converter
    stores `chi` and `sigma_rupture` as
    [`Property`][jbubble.bubble.property.Property] instances.

    References
    ----------
    Gümmer, J., Schenke, S., & Denner, F. (2021). Modelling lipid-coated
    microbubbles in focused ultrasound applications at subresonance
    frequencies. *Ultrasound Med. Biol.* 47(10), 2958-2979.
    [doi:10.1016/j.ultrasmedbio.2021.06.012](https://doi.org/10.1016/j.ultrasmedbio.2021.06.012)
    """

    R_buckle_ratio: ArrayLike
    chi: Property = eqx.field(converter=as_property)
    sigma_rupture: Property = eqx.field(converter=as_property)

    def __check_init__(self) -> None:
        ratio = _concrete_value(self.R_buckle_ratio)
        chi = _concrete_value(self.chi)
        sigma_r = _concrete_value(self.sigma_rupture)
        if ratio is None or chi is None or sigma_r is None:
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_0 = chi * ((1.0 / ratio) ** 2 - 1.0)
        well_posed = (ratio > 0.0) & (chi > 0.0) & (sigma_0 > 0.0) & (sigma_0 < sigma_r)
        if not np.all(well_posed):
            raise ValueError(
                "GompertzSurfaceTension is ill-posed: the surface tension at R0, "
                "sigma_0 = chi * ((1 / R_buckle_ratio)**2 - 1) = "
                f"{_format_values(sigma_0)} N/m, must lie strictly between 0 and "
                f"sigma_rupture = {_format_values(sigma_r)} N/m, with chi > 0. "
                "Use 0 < R_buckle_ratio < 1 with "
                "chi * ((1 / R_buckle_ratio)**2 - 1) < sigma_rupture, or use "
                "SmoothMarmottantSurfaceTension, which has no such constraint."
            )

    def __call__(self, state: BubbleState) -> jax.Array:
        R, R0 = state.R, state.R0
        R_b = self.R_buckle_ratio * R0
        chi = self.chi(state)
        sigma_r = self.sigma_rupture(state)
        c = (2.0 * jnp.e * chi / sigma_r) * jnp.sqrt(1.0 + sigma_r / (2.0 * chi))
        # ln(sigma_0 / sigma_r), negative for a well-posed law.
        log_q = jnp.log(chi * ((R0 / R_b) ** 2 - 1.0) / sigma_r)
        # Past an exponent of 50, sigma is zero in float64. The cap keeps exp()
        # and its gradient finite under deep compression.
        expo = jnp.minimum(c * (R0 - R) / R_b, 50.0)
        return sigma_r * jnp.exp(log_q * jnp.exp(expo))


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
