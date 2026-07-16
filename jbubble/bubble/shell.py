"""Shell / coating models.

Computes the total inward stress from the bubble shell, including
Laplace pressure from surface tension, viscous dissipation, and
elastic restoring forces.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .property import Property, as_property
from .state import BubbleState, HystereticShellState


class ShellModel(eqx.Module, abc.ABC):
    """Bubble shell / coating model.

    Computes the total inward stress from the shell, including:

    - Laplace pressure from surface tension:  2 sigma(R) / R
    - Shell viscous dissipation  (e.g. 4 kappa_s Rdot / R^2)
    - Shell elastic restoring forces  (for thick shells)

    Every ``ShellModel`` holds a ``Property`` as its ``sigma`` field.
    A plain float is accepted and auto-converted to a ``Property``
    in ``__post_init__``.

    """

    sigma: Property = eqx.field(converter=as_property)

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

    # ── optional multi-physics hooks (default: no extra state) ───────────────
    # A shell with its own dynamics (e.g. an evolving integrity variable)
    # overrides these; the EoM calls them generically, mirroring GasModel.

    def augment_initial_state(self, base: BubbleState) -> BubbleState:
        """Return the initial state extended with any shell-owned DOFs.

        Default: pass through.
        """
        return base

    def d_state(self, state: BubbleState) -> BubbleState:
        """Shell-owned contribution to d(state)/dt (zero everywhere except the
        shell's own dynamic fields).  Default: all zeros (a memoryless shell)."""
        return jax.tree_util.tree_map(jnp.zeros_like, state)


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

    Note on elastic contributions
    -----------------------------
    ``p_elastic`` returns zero for this model.  The shell elasticity is
    not absent — it is encoded entirely in the surface tension Property
    ``sigma``.  When ``sigma`` is state-dependent (e.g.
    ``MarmottantSurfaceTension``, ``GompertzSurfaceTension``), the
    area-elasticity term χ((R/Rb)² − 1) enters through ``p_laplace =
    2 σ(R) / R``, not through a separate ``p_elastic`` term.  This is
    consistent with how the Marmottant model is written in the
    literature: the elastic and ruptured regimes modify σ(R) rather than
    adding an independent stress contribution.

    Fields
    ------
    sigma : float or Property
        Surface tension law.
    kappa_s : float or Property
        Shell surface-dilatational viscosity  [N s/m].
    """

    kappa_s: Property = eqx.field(converter=as_property)

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.kappa_s(state) * state.R_dot / state.R**2


class HystereticShell(LipidShell):
    """Lipid shell with an irreversible-rupture integrity state (hysteresis).

    A single-valued σ(R) law (Marmottant / Gompertz) is *reversible*: the σ–R
    curve is retraced identically on expansion and compression, so it encloses
    no area and dissipates no energy.  Real lipid monolayers instead **rupture /
    shed** when over-stretched and only slowly re-form, so the effective tension
    depends on loading history — an area-enclosing (dissipative) σ–R loop.

    This model resolves that history with a shell-integrity state variable
    ``phi`` ∈ [0, 1] on :class:`~jbubble.bubble.state.HystereticShellState`
    (1 = intact, 0 = ruptured).  The effective surface tension interpolates
    between the intact law and the bare/ruptured tension::

        σ_eff(R, φ) = φ · σ_intact(R)  +  (1 − φ) · σ_rupture

    so an intact shell (φ = 1) reproduces the wrapped law exactly, while a
    ruptured shell (φ = 0) behaves like a bare bubble at σ_rupture.  The
    integrity evolves as::

        φ̇ = − k_rup · φ · relu(R/R0 − r_rup)   +   k_heal · (1 − φ)

    The first term drops φ (irreversibly, since it ∝ φ) once the bubble expands
    past ``r_rup·R0``; the second re-forms the monolayer at rate ``k_heal``.
    Because the drop happens at large R and the recovery lags, the return stroke
    sees a different σ than the outward stroke → a hysteresis loop whose enclosed
    area is the per-cycle dissipated energy.  The loss is **amplitude-dependent**:
    small cycles never cross ``r_rup`` and dissipate nothing; large cycles rupture
    and dissipate strongly.  The φ dynamics are self-bounding in [0, 1] (the
    rupture term ∝ φ, the healing term ∝ (1 − φ)), so no clamping is needed.

    Limit: ``k_rup → 0`` (or expansion never reaching ``r_rup``) leaves φ ≡ 1 and
    recovers the wrapped intact shell exactly.

    Fields
    ------
    sigma : Property
        Intact surface-tension law (e.g. ``GompertzSurfaceTension``).
    kappa_s : Property
        Shell surface-dilatational viscosity  [N s/m]  (inherited).
    sigma_rupture : float
        Bare / ruptured surface tension  [N/m]  (the φ = 0 asymptote).
    k_rup : float or Property
        Rupture rate  [1/s]  (per unit dimensionless over-expansion).
    r_rup : float or Property
        Expansion ratio R/R0 at which rupture sets in  (dimensionless).
    k_heal : float or Property
        Monolayer re-formation (healing) rate  [1/s].
    """

    sigma_rupture: float
    k_rup: Property = eqx.field(converter=as_property)
    r_rup: Property = eqx.field(converter=as_property)
    k_heal: Property = eqx.field(converter=as_property)

    def _phi(self, state: BubbleState) -> jax.Array:
        # intact (φ=1) for a plain BubbleState (e.g. equilibrium seeding);
        # the dynamic integrity for a HystereticShellState.
        return getattr(state, "phi", jnp.ones_like(state.R))

    def sigma_eff(self, state: BubbleState) -> jax.Array:
        phi = self._phi(state)
        return phi * self.sigma(state) + (1.0 - phi) * self.sigma_rupture

    def p_laplace(self, state: BubbleState) -> jax.Array:
        return 2.0 * self.sigma_eff(state) / state.R

    def _dphi_dt(self, state: BubbleState) -> jax.Array:
        # hard threshold: rupture driver is exactly zero below r_rup, so the
        # resting shell (R = R0 < r_rup·R0) is a true fixed point at φ = 1.
        over = jax.nn.relu(state.R / state.R0 - self.r_rup(state))
        phi = state.phi  # ty: ignore[unresolved-attribute]
        return -self.k_rup(state) * phi * over + self.k_heal(state) * (1.0 - phi)

    def augment_initial_state(self, base: BubbleState) -> HystereticShellState:
        return HystereticShellState(
            R=base.R,
            R_dot=base.R_dot,
            R0=base.R0,
            P_gas0=base.P_gas0,
            phi=jnp.ones_like(base.R),
        )

    def d_state(self, state: BubbleState) -> BubbleState:
        zero = jax.tree_util.tree_map(jnp.zeros_like, state)
        return eqx.tree_at(
            lambda s: s.phi,  # ty: ignore[unresolved-attribute]
            zero,
            self._dphi_dt(state),
        )


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
    """Piecewise Marmottant surface tension law.

    Three regimes based on the ratio R / R_buckle::

        R <= R_buckle                :  sigma = 0               (buckled)
        R_buckle < R < R_rupture     :  sigma = chi ((R/R_b)^2 - 1)  (elastic)
        R >= R_rupture               :  sigma = sigma_rupture   (ruptured)

    where R_buckle = R_buckle_ratio * state.R0 and R_rupture is derived
    from continuity of sigma at the elastic-to-ruptured transition.

    Note: sigma(R) has discontinuous first derivatives at the regime
    boundaries.  For applications requiring smooth gradients (e.g.
    gradient-based optimisation), use ``GompertzSurfaceTension`` instead.

    Fields
    ------
    R_buckle_ratio : float
        Buckling radius as a fraction of R0  (dimensionless).
    chi : float or Property
        Shell elasticity  [N/m].
    sigma_rupture : float or Property
        Surface tension (post-rupture value)  [N/m].
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
    """Smooth Gompertz surface tension law.

    A differentiable Gompertz function approximates the piecewise
    Marmottant surface tension, enabling robust automatic
    differentiation::

        sigma(R) = a exp(-b exp(c (1 - R / R_buckle)))

    where R_buckle = R_buckle_ratio * state.R0.

    The Gompertz parameters b and c are derived from chi and sigma_rupture
    such that sigma(R0) matches the elastic regime and sigma -> sigma_rupture
    as R -> infinity.  R0 is read from the state so this model stays
    consistent when R0 evolves (e.g. rectified diffusion).

    Steepness of the transition is set by c, which is scaled by the
    dimensionless ``sharpness`` factor.  The base c (sharpness = 1) matches
    the Marmottant *elastic slope at R0*, which gives a poor global fit to
    the piecewise curve.  The default ``sharpness = 3.3`` is the mean
    least-squares optimum over the elastic regime across the typical
    parameter range, giving a markedly tighter match to Marmottant.
    Scaling c leaves both anchors fixed: b is re-solved so sigma(R0) is
    unchanged, and sigma -> sigma_rupture as R -> infinity regardless.

    Well-posedness constraint
    -------------------------
    The Gompertz fit requires that the initial surface tension at R0 lies
    strictly below the rupture threshold::

        chi * ((1 / R_buckle_ratio)^2 - 1) < sigma_rupture

    Construction raises ``ValueError`` if this is violated.  A common
    mistake is setting R_buckle_ratio too small (e.g. 0.9), which inflates
    sigma(R0) above sigma_rupture.  Values around 0.95–0.99 are typical.

    Fields
    ------
    R_buckle_ratio : float
        Buckling radius as a fraction of R0  (dimensionless).
    chi : float or Property
        Shell elasticity  [N/m].
    sigma_rupture : float or Property
        Asymptotic (ruptured) surface tension  [N/m].
    sharpness : float
        Dimensionless multiplier on the transition rate c.  Default 3.3
        (least-squares fit to Marmottant); 1.0 recovers the elastic-slope
        match at R0.
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
