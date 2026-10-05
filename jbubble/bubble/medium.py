"""Surrounding medium (fluid or tissue) models.

Each model computes the total inward viscous and elastic stresses that
the surrounding medium exerts on the bubble wall.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .property import Property, as_property
from .state import BubbleState

__all__ = [
    "MediumModel",
    "NewtonianMedium",
    "KelvinVoigtMedium",
    "NeoHookeanMedium",
    "PowerLawMedium",
]


class MediumModel(eqx.Module, abc.ABC):
    r"""Surrounding medium (fluid or tissue) model.

    Computes the total inward viscous and elastic stresses that the
    surrounding medium exerts on the bubble wall. For a Newtonian liquid
    this is $4\mu\dot{R}/R$. Viscoelastic media, such as Kelvin-Voigt and
    neo-Hookean media, add elastic restoring forces.

    Subclasses implement separate methods for the viscous and elastic
    contributions, and the default `__call__` sums them to get the total
    medium pressure $p_\text{medium}(\text{state})$.

    The `mu` field's [`as_property`][jbubble.bubble.property.as_property]
    converter turns a plain float into a
    [`Property`][jbubble.bubble.property.Property].

    Parameters
    ----------
    mu : float or Property
        Viscous scaling parameter. For Newtonian, Kelvin-Voigt, and
        neo-Hookean media this is the dynamic viscosity [Pa s]. For
        [`PowerLawMedium`][jbubble.bubble.medium.PowerLawMedium] it is the
        consistency index $K$ [Pa sⁿ], which equals the dynamic viscosity
        when $n = 1$.
    """

    mu: Property = eqx.field(converter=as_property)

    @abc.abstractmethod
    def p_viscous(self, state: BubbleState) -> jax.Array:
        """Return the viscous contribution to the medium pressure [Pa]."""
        ...

    @abc.abstractmethod
    def p_elastic(self, state: BubbleState) -> jax.Array:
        """Return the elastic contribution to the medium pressure [Pa]."""
        ...

    def __call__(self, state: BubbleState) -> jax.Array:
        r"""Compute the total medium pressure, $p_\text{medium}(\text{state})$.

        Parameters
        ----------
        state : BubbleState
            Current bubble state.

        Returns
        -------
        jax.Array
            Scalar total inward medium pressure (viscous plus elastic) [Pa].
        """
        return self.p_viscous(state) + self.p_elastic(state)


class NewtonianMedium(MediumModel):
    r"""Newtonian liquid medium.

    $$
    p_\text{medium} = \frac{4\mu\dot{R}}{R}
    $$

    Parameters
    ----------
    mu : float or Property
        Dynamic viscosity [Pa s].
    """

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu(state) * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0


class KelvinVoigtMedium(MediumModel):
    r"""Kelvin-Voigt viscoelastic medium (Yang & Church 2005).

    $$
    p_\text{medium} = \frac{4\mu\dot{R}}{R}
        + \frac{4G}{3}\left[1 - \left(\frac{R_0}{R}\right)^3\right]
    $$

    The elastic term integrates a linear (Hookean) elastic stress through
    the incompressible surrounding solid. Yang and Church take the
    displacement field $u(r) = (R^3 - R_0^3)/(3r^2)$ of the incompressible
    radial motion, so the deviatoric stress difference is
    $\tau_{rr} - \tau_{\theta\theta} = -2G(R^3 - R_0^3)/r^3$, and

    $$
    p_\text{elastic} = -2\int_R^\infty
        \frac{\tau_{rr} - \tau_{\theta\theta}}{r}\,\mathrm{d}r
        = \frac{4G}{3}\left[1 - \left(\frac{R_0}{R}\right)^3\right].
    $$

    The model reads $R_0$ from `state.R0`. The elastic term is zero at
    $R = R_0$, linearises to $4G(R - R_0)/R_0$, which is the same small-strain
    limit as [`NeoHookeanMedium`][jbubble.bubble.medium.NeoHookeanMedium],
    and stays bounded by $4G/3$ under expansion. For large strains, prefer
    the neo-Hookean model, whose constitutive law holds at finite strain.

    Parameters
    ----------
    mu : float or Property
        Dynamic viscosity [Pa s].
    G : float or Property
        Shear modulus [Pa]. It can be state-dependent, for example to model
        strain stiffening.

    References
    ----------
    Yang, X., & Church, C. C. (2005). A model for the dynamics of gas
    bubbles in soft tissue. *The Journal of the Acoustical Society of
    America*, 118(6), 3595-3606. <https://doi.org/10.1121/1.2118307>

    Warnez, M. T., & Johnsen, E. (2015). Numerical modeling of bubble
    dynamics in viscoelastic media with relaxation. *Physics of Fluids*,
    27(6), 063103, Eq. 42. <https://doi.org/10.1063/1.4922598>
    """

    G: Property = eqx.field(converter=as_property)

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu(state) * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return (4.0 / 3.0) * self.G(state) * (1.0 - (state.R0 / state.R) ** 3)


class NeoHookeanMedium(MediumModel):
    r"""Neo-Hookean finite-strain viscoelastic medium.

    Captures finite-strain behaviour at large oscillation amplitudes, where
    the Kelvin-Voigt linear approximation breaks down.

    The elastic pressure comes from integrating the deviatoric stress
    difference $\sigma_{\theta\theta} - \sigma_{rr}$ through the
    surrounding incompressible neo-Hookean solid. With the substitution
    $v = r_0/r$, the integral collapses exactly to
    $2G\int_{R_0/R}^{1} (1 + v^3)\,\mathrm{d}v$, which gives

    $$
    p_\text{elastic} = G\left[\frac{5}{2} - 2\frac{R_0}{R}
        - \frac{1}{2}\left(\frac{R_0}{R}\right)^4\right].
    $$

    Key behaviour:

    - Zero at $R = R_0$: no elastic stress at equilibrium.
    - Reduces to $4G(R - R_0)/R_0$ for small strains, identical to
      Kelvin-Voigt.
    - Saturates to $5G/2$ as $R \to \infty$. Physically, the material
      near the bubble thins to zero, so the elastic pressure plateaus.
    - Diverges to $-\infty$ as $R \to 0$: strong restoring force under
      compression.

    The total medium pressure is

    $$
    p_\text{medium} = \frac{4\mu\dot{R}}{R}
        + G\left[\frac{5}{2} - 2\frac{R_0}{R}
        - \frac{1}{2}\left(\frac{R_0}{R}\right)^4\right].
    $$

    Parameters
    ----------
    mu : float or Property
        Dynamic viscosity [Pa s]. Accepts a plain float.
    G : float or Property
        Shear modulus [Pa]. Accepts a plain float. It can be
        state-dependent, for example to model strain stiffening.
    """

    G: Property = eqx.field(converter=as_property)

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu(state) * state.R_dot / state.R

    def p_elastic(self, state: BubbleState) -> jax.Array:
        beta = state.R0 / state.R  # R0/R
        return self.G(state) * (2.5 - 2.0 * beta - 0.5 * beta**4)


class PowerLawMedium(MediumModel):
    r"""Power-law (generalised Newtonian) surrounding medium.

    The viscosity depends on the local shear rate,
    $\eta = K\dot{\gamma}^{\,n-1}$, where
    $\dot{\gamma} = \sqrt{2\mathbf{D}:\mathbf{D}}$ is the rheometric
    shear rate, the convention under which rheometers report $K$ and $n$.
    For the incompressible radial flow $u = R^2\dot{R}/r^2$ the shear rate
    is $\dot{\gamma}(r) = 2\sqrt{3}\,|R^2\dot{R}|/r^3$, which is
    $\dot{\gamma}_w = 2\sqrt{3}\,|\dot{R}|/R$ at the bubble wall.
    Integrating the viscous stress through the liquid,

    $$
    p_\text{viscous} = -2\int_R^\infty
        \frac{\tau_{rr} - \tau_{\theta\theta}}{r}\,\mathrm{d}r
        = \frac{4K}{n}\,\dot{\gamma}_w^{\,n-1}\,\frac{\dot{R}}{R}.
    $$

    The factor $1/n$ comes from the radial variation of $\eta$. For
    $n = 1$ this is exactly the Newtonian result $4K\dot{R}/R$.

    The pure power law has an infinite viscosity at zero shear rate when
    $n < 1$, which makes the right-hand side non-smooth whenever
    $\dot{R}$ changes sign. The model regularises the wall shear rate
    smoothly:

    $$
    \dot{\gamma}_\text{eff}
        = \sqrt{12\left(\frac{\dot{R}}{R}\right)^2 + \varepsilon^2},
    \qquad
    p_\text{viscous} = \frac{4K}{n}\,\dot{\gamma}_\text{eff}^{\,n-1}\,
        \frac{\dot{R}}{R}.
    $$

    Special cases:

    - $n < 1$: shear-thinning (biological tissue, blood, polymer gels).
    - $n = 1$: Newtonian; recovers
      [`NewtonianMedium`][jbubble.bubble.medium.NewtonianMedium] with
      viscosity $K$.
    - $n > 1$: shear-thickening.

    Parameters
    ----------
    mu : float or Property
        Consistency index $K$ [Pa sⁿ], in the rheometric convention
        $\eta = K\dot{\gamma}^{\,n-1}$ with
        $\dot{\gamma} = \sqrt{2\mathbf{D}:\mathbf{D}}$. The inherited field
        name is `mu`, but its dimensions are [Pa sⁿ], not [Pa s]. It equals
        the dynamic viscosity when $n = 1$.
    n_exp : float or Property
        Power-law exponent (dimensionless, positive).
    eps : float
        Shear-rate regularisation $\varepsilon$ [s⁻¹]. Default: `1e2`.

    Notes
    -----
    The default $\varepsilon = 10^2$ s⁻¹ is three or more orders of
    magnitude below the wall shear rates of a driven microbubble (a 1 %
    oscillation at 1 MHz gives
    $\dot{\gamma}_w \approx 2\sqrt{3}\,(2\pi \cdot 10^6)(0.01)
    \approx 2 \times 10^5$ s⁻¹), so it changes the viscous pressure only
    near the instants where $\dot{R}$ reverses. For a 2 µm bubble in a
    shear-thinning liquid ($n = 0.7$) at 100 kPa, the radius differs from
    the $\varepsilon \to 0$ solution by less than $10^{-7} R_0$. A much
    smaller $\varepsilon$ (for example the `1e-6` of jbubble 0.1) leaves
    a near-singular derivative at $\dot{R} = 0$, on which tight-tolerance
    solves fail at 300 kPa. If you lower $\varepsilon$, check that your
    solve still converges.

    The earlier jbubble convention used the wall velocity gradient
    $2|\dot{R}|/R$ as the shear rate, which overestimates the viscous
    pressure by $\sqrt{3}^{\,1-n}$ for a $K$ measured on a rheometer.

    References
    ----------
    Kaykanat, S. I., & Uguz, K. (2024). Shape stability of a microbubble in
    a power-law liquid. *The European Physical Journal Special Topics*,
    Eq. 17, with $M = 2m(2\sqrt{3})^{k-1}$.
    <https://doi.org/10.1140/epjs/s11734-024-01174-7>
    """

    n_exp: Property = eqx.field(converter=as_property)
    eps: float = 1e2

    def p_viscous(self, state: BubbleState) -> jax.Array:
        n_val = self.n_exp(state)
        K_val = self.mu(state)
        strain_rate = state.R_dot / state.R
        gamma_dot = jnp.sqrt(12.0 * strain_rate**2 + self.eps**2)
        return 4.0 * K_val / n_val * gamma_dot ** (n_val - 1.0) * strain_rate

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0
