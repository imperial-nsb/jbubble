r"""Diffrax-based ODE solvers for bubble dynamics.

[`solve_eom`][jbubble.solver.solve_eom] integrates an
[`EquationOfMotion`][jbubble.bubble.eom.EquationOfMotion] with the settings
in [`SolverConfig`][jbubble.solver.SolverConfig] and samples the solution as
[`SaveSpec`][jbubble.solver.SaveSpec] describes.

Two details matter when you choose tolerances or differentiate through a
solve:

- **Scaled state.** The solver integrates the dimensionless state
  `state / eom.state_scale(state)`, where the radius is in units of `R0` and
  the wall velocity in units of $\sqrt{P_\text{amb}/\rho_L}$. The step-size
  controller's `rtol` and `atol` therefore apply to that dimensionless state,
  and mean the same thing for every bubble size.
- **Guarded right-hand side.** The equation of motion is never evaluated at
  a state outside its domain, such as `R <= 0`, so gradients stay finite
  when an adaptive solver tries, and then rejects, a step that overshoots a
  violent collapse. With an explicit solver, such as the default `Dopri5`,
  the forward solution is the same, bit for bit, as without the guard.
"""

from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
from jax.typing import ArrayLike

from .bubble.eom import EquationOfMotion
from .pulse import Pulse

__all__ = ["SaveSpec", "SolverConfig", "solve_eom"]


def _step_to_window_edges(
    controller: diffrax.AbstractStepSizeController, pulse: Pulse
) -> diffrax.AbstractStepSizeController:
    """Make an adaptive `controller` step to every window edge of `pulse`.

    While the drive is zero, before a delayed pulse starts, an adaptive
    controller grows its step size by up to 10 times per step. Without
    these step times, it can step over a delayed pulse entirely, so the
    bubble never moves. The edges are
    [`Pulse.window_edges`][jbubble.pulse.base.Pulse.window_edges], with
    gradients stopped, because a step location carries no meaningful
    derivative. Other controllers, such as `diffrax.ConstantStepSize`, are
    returned unchanged.
    """
    if not isinstance(controller, diffrax.AbstractAdaptiveStepSizeController):
        return controller
    edges = jax.lax.stop_gradient(pulse.window_edges)
    return diffrax.ClipStepSizeController(controller, step_ts=edges)


class SaveSpec(eqx.Module):
    """Specification for ODE output sampling.

    Parameters
    ----------
    num_samples : int
        Number of evenly spaced time points to record. Default: `1024`.
    """

    num_samples: int = eqx.field(default=1024, static=True)

    def build(self, t0: jax.Array, t1: jax.Array) -> diffrax.SaveAt:
        """Return a `diffrax.SaveAt` with `num_samples` times from `t0` to `t1`.

        Parameters
        ----------
        t0 : jax.Array
            First saved time [s].
        t1 : jax.Array
            Last saved time [s].

        Returns
        -------
        diffrax.SaveAt
            Save specification for `diffrax.diffeqsolve`.
        """
        ts = jnp.linspace(t0, t1, self.num_samples)
        return diffrax.SaveAt(ts=ts)


_DEFAULT_RTOL = 1e-6
_DEFAULT_ATOL = 1e-10
_DEFAULT_MAX_STEPS = 100_000
# The smallest absolute tolerance of the Newton iterations in
# SolverConfig.stiff. Kvaerno5 solves for the stage derivatives of the scaled
# state, and optimistix.Chord compares the Newton residual with atol alone,
# in units of the scaled state per second. Roundoff in a nanobubble's
# right-hand side keeps that residual above about 1e-8, so a smaller floor
# stalls 10 to 40 nm bubbles; 1e-6 converges for all of them.
_NEWTON_ATOL_FLOOR = 1e-6


class SolverConfig(eqx.Module):
    r"""Numerical integration settings for [`solve_eom`][jbubble.solver.solve_eom].

    The defaults suit microbubbles in water: an explicit fifth-order
    Runge-Kutta method (`diffrax.Dopri5`) with an adaptive step size. For
    stiff problems, use [`SolverConfig.stiff`][jbubble.solver.SolverConfig.stiff].

    Parameters
    ----------
    solver : diffrax.AbstractSolver
        ODE solver. Default: `diffrax.Dopri5()`.
    stepsize_controller : diffrax.AbstractStepSizeController
        Step-size controller. Default:
        `diffrax.PIDController(rtol=1e-6, atol=1e-10)`. The tolerances apply
        to the scaled state of [`solve_eom`][jbubble.solver.solve_eom], so
        `atol=1e-10` means $10^{-10} R_0$ for the radius and
        $10^{-10}\sqrt{P_\text{amb}/\rho_L}$, about $10^{-9}$ m/s in water,
        for the wall velocity.
    dt0 : float
        Initial step size [s]. Default: `1e-9`.
    max_steps : int
        Maximum number of solver steps per integration. Default: `100_000`.
        A solve that needs more steps stops early and reports
        `converged = False`.

    Notes
    -----
    **Choosing tolerances.** At the defaults, a microbubble driven at
    100 kPa has radius errors of $10^{-6}$ to $10^{-5} R_0$ and relative
    gradient errors of $10^{-5}$ to $10^{-3}$. The kink in a Marmottant
    surface-tension law is the hardest case, with radius errors up to about
    $10^{-3} R_0$ and gradient errors of a few percent, or about 12 % for the
    peak radius of the `lipid_bubble` preset with respect to `kappa_s`. Use
    tighter tolerances when you fit a lipid shell. Through an
    inertial collapse ($R_\text{max}/R_0 \approx 4$), both errors are about
    $10^{-3}$ to $10^{-2}$. `rtol=1e-8, atol=1e-12` makes them 100 to 1000
    times smaller for two to three times as many steps. Looser tolerances
    can make gradients through a collapse wrong by more than 100 %.

    `rtol` sets the accuracy of most solves. `atol` matters where a scaled
    state component is far below 1: the wall velocity of a small or weakly
    driven bubble, which can be millimetres per second. The default
    velocity tolerance, about $10^{-9}$ m/s, matches jbubble 0.1. Unlike in
    0.1, where `atol=1e-9` applied in metres, the radius tolerance is
    relative to $R_0$, so a collapsing microbubble no longer has a loose
    radius tolerance: through an inertial collapse, 0.1's gradients could be
    off by 50 % or more.

    **Stiffness.** A bubble is stiff when its fastest relaxation rate is
    much larger than the driving angular frequency $\omega$. For a viscous
    liquid and a lipid shell of dilatational viscosity $\kappa_s$, that rate
    is about

    $$
    \lambda = \frac{4\mu + 4\kappa_s/R_0}{\rho_L R_0^2},
    $$

    divided by $1 + (4\mu + 4\kappa_s/R_0)/(\rho_L c_L R_0)$ for
    Keller-Miksis. The problem is stiff when $\lambda/\omega \gtrsim 100$.
    For example, a lipid shell with $\kappa_s = 7.5 \times 10^{-9}$ kg/s
    driven at 5 MHz gives $\lambda/\omega \approx 850$ at 50 nm, 320 at
    100 nm, and 150 at 150 nm, and a 300 nm bubble in a 0.05 Pa s liquid at
    2 MHz gives 120. A 2 µm lipid-coated microbubble at 1 MHz gives 0.8.

    On a stiff problem, `Dopri5` steps at the edge of its stability region:
    it needs hundreds to thousands of steps per driving period, and
    although the radius stays accurate, at the default tolerances the
    gradient through the solve can be wrong by orders of magnitude, or NaN,
    even when the solve reports `converged = True`: a 20 nm lipid bubble
    driven at 5 MHz gives a NaN gradient. Use
    [`stiff`][jbubble.solver.SolverConfig.stiff] there.

    **Long integrations.** `Dopri5` needs about 60 to 100 steps, accepted
    and rejected, per driving period for a microbubble in water, so
    `max_steps=100_000` covers about a thousand periods. Raise it for
    longer pulses. A larger `max_steps` costs neither memory nor compile
    time under `jax.grad`.
    """

    solver: diffrax.AbstractSolver = eqx.field(default_factory=diffrax.Dopri5)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        default_factory=lambda: diffrax.PIDController(
            rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL
        )
    )
    dt0: float = 1e-9
    max_steps: int = eqx.field(default=_DEFAULT_MAX_STEPS, static=True)

    @classmethod
    def stiff(
        cls,
        *,
        rtol: float = _DEFAULT_RTOL,
        atol: float = _DEFAULT_ATOL,
        dt0: float = 1e-9,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> SolverConfig:
        r"""Return settings for stiff problems: an implicit `Kvaerno5` solver.

        Use this configuration for nanobubbles, sub-micron bubbles in
        viscous liquids, and other stiff problems (see the Notes of
        [`SolverConfig`][jbubble.solver.SolverConfig]). It returns

        ```python
        SolverConfig(
            solver=diffrax.Kvaerno5(
                root_finder=optimistix.Chord(
                    rtol=rtol, atol=max(atol, 1e-6), norm=optimistix.rms_norm
                )
            ),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            dt0=dt0,
            max_steps=max_steps,
        )
        ```

        Two choices make the implicit solver reliable:

        - **The `Chord` root finder.** `Kvaerno5`'s default root finder,
          `diffrax.VeryChord`, reports a Newton iteration that converges
          slowly as diverged in diffrax 0.7.2 with optimistix 0.1.0, so the
          solver rejects many steps: on a moderately stiff or a non-stiff
          problem it takes 6 to 13 times more steps than with `Chord`.
          [Diffrax pull request 754](https://github.com/patrick-kidger/diffrax/pull/754)
          fixes this, but no release contains it yet.
        - **A floor on the Newton tolerance.** `Kvaerno5` solves for the
          derivative of the scaled state at each stage, and `Chord` compares
          the residual of those equations, in units of the scaled state per
          second, with its absolute tolerance alone. A nanobubble's
          right-hand side is a sum of large terms that cancel, so roundoff
          keeps that residual above about `1e-8`. With a smaller
          tolerance, the solver rejects most steps of a 10 to 40 nm bubble
          and stops at `max_steps`. The Newton iterations therefore use an
          absolute tolerance of at least `1e-6`.

        Parameters
        ----------
        rtol : float
            Relative tolerance of the step-size controller and of the Newton
            iterations. Default: `1e-6`.
        atol : float
            Absolute tolerance of the step-size controller, on the scaled
            state (see [`SolverConfig`][jbubble.solver.SolverConfig]).
            Default: `1e-10`. The Newton iterations use `max(atol, 1e-6)`.
        dt0 : float
            Initial step size [s]. Default: `1e-9`.
        max_steps : int
            Maximum number of solver steps. Default: `100_000`.

        Returns
        -------
        SolverConfig
            Implicit-solver settings.

        Notes
        -----
        On a stiff problem, `Kvaerno5` takes 5 to 50 times fewer steps than
        `Dopri5`, or converges where `Dopri5` reaches `max_steps`, and its
        gradients are accurate. For 20 to 150 nm lipid bubbles at the
        default tolerances, the gradient of a loss on the deviation
        $R/R_0 - 1$ with respect to the shell viscosity has a relative error
        of about $10^{-5}$ to $10^{-4}$. A loss dominated by the mean radius,
        such as one on $R/R_0$, has larger errors: about $10^{-3}$ at 20 nm
        and $10^{-2}$ at 10 nm. On a problem that isn't stiff, it takes
        about as many steps as `Dopri5`, with similar errors, but runs 5 to
        10 times longer. It also compiles more slowly: about 2 to 6 s for a
        solve, against about 1 s or less for `Dopri5`.

        To change the tolerances, call `stiff` again with new `rtol` and
        `atol`, rather than replacing `stepsize_controller`, so that the
        Newton tolerances follow.

        `jax.jit`, `jax.vmap`, and `jax.grad` work with this configuration.
        `jax.pmap` doesn't: with diffrax 0.7.2, an implicit solve under
        `jax.pmap` raises `ValueError: pytree does not match out_structure`.

        Examples
        --------
        ```python
        from jbubble import SolverConfig, run_simulation

        result = run_simulation(eom, pulse, config=SolverConfig.stiff())
        ```
        """
        root_finder = optx.Chord(
            rtol=rtol, atol=max(atol, _NEWTON_ATOL_FLOOR), norm=optx.rms_norm
        )
        return cls(
            solver=diffrax.Kvaerno5(root_finder=root_finder),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            dt0=dt0,
            max_steps=max_steps,
        )


def _is_admissible(eom: EquationOfMotion, state: Any) -> jax.Array:
    """`True` if every state leaf is finite and the state is in the EoM's domain."""
    finite = jnp.stack([jnp.all(jnp.isfinite(x)) for x in jtu.tree_leaves(state)])
    return jnp.all(finite) & jnp.all(eom.is_admissible(state))


def _guarded_vector_field(t: Any, z: Any, args: tuple) -> Any:
    """Right-hand side for the scaled state `z = state / scale`, NaN-safe.

    An adaptive solver tries trial steps that it later rejects. During a
    strong collapse, a stage of such a trial step can reach `R <= 0`, or
    inherit a NaN from an earlier stage. There the gas law evaluates
    `(R0 / R) ** (3 * gamma)` of a negative base, so the right-hand side and
    its Jacobian are NaN. Diffrax rejects the step, so the solution is
    unaffected, but reverse-mode autodiff still multiplies the rejected
    step's NaN Jacobian by a zero cotangent, and `0 * NaN` is NaN.

    This function applies the "double `where`" from the JAX FAQ:

        ok     = all_finite(z * scale) and eom.is_admissible(z * scale)
        z_safe = where(ok, z, z_ref)
        dz     = where(ok, eom(t, z_safe * scale, pulse) / scale, NaN)

    The equation of motion is always evaluated at an admissible state, so
    its Jacobian is finite, and the output is still NaN at an inadmissible
    state, so the solver still rejects the step. At an admissible state the
    result is the same, bit for bit, as without the guard,
    `eom(t, z * scale, pulse) / scale`. Two details keep it so:

    - The guard selects `z` before the multiplication by `scale`, not the
      SI state after it. A select between a multiplication and the addition
      that consumes it stops XLA from fusing the pair into one fused
      multiply-add, which rounds differently.
    - The output select comes after the division by `scale`, which keeps
      XLA from rewriting `(a / b) / c` as `a / (b * c)` in one version only.
    """
    eom, pulse, z_ref, scale = args
    ok = _is_admissible(eom, jtu.tree_map(jnp.multiply, z, scale))
    z_safe = jtu.tree_map(lambda a, b: jnp.where(ok, a, b), z, z_ref)
    d_state = eom(t, jtu.tree_map(jnp.multiply, z_safe, scale), pulse)
    dz = jtu.tree_map(jnp.divide, d_state, scale)
    return jtu.tree_map(lambda d: jnp.where(ok, d, jnp.nan), dz)


def _as_inexact(x: Any) -> jax.Array:
    """Return `x` as an array, promoting an integer or boolean dtype to float."""
    x = jnp.asarray(x)
    return x.astype(jnp.result_type(x, float))


def _with_equilibrium(eom: EquationOfMotion, y0: Any) -> Any:
    """Fill a zero (unset) `R0` or `P_gas0` in `y0` from the EoM.

    A zero `R0` becomes `eom.R0`, and a zero `P_gas0` becomes the
    Laplace-equilibrium gas pressure for the (filled) `R0`. An integer field,
    such as `R0=0`, becomes floating point first, so the fill isn't truncated.
    """
    R0_in, P_gas0_in = _as_inexact(y0.R0), _as_inexact(y0.P_gas0)
    R0 = jnp.where(R0_in == 0, jnp.asarray(eom.R0, dtype=R0_in.dtype), R0_in)
    equilibrium = eqx.tree_at(lambda e: e.R0, eom, R0).initial_state()
    P_gas0 = jnp.where(P_gas0_in == 0, equilibrium.P_gas0, P_gas0_in)
    return eqx.tree_at(lambda s: (s.R0, s.P_gas0), y0, (R0, P_gas0))


def solve_eom(
    eom: EquationOfMotion,
    pulse: Pulse,
    *,
    y0: Any = None,
    t_max: ArrayLike | None = None,
    save_spec: SaveSpec | None = None,
    config: SolverConfig | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    progress: bool = False,
) -> diffrax.Solution:
    r"""Solve the bubble dynamics for an equation of motion.

    Integrates the initial value problem

    $$
    \frac{\mathrm{d}\,\text{state}}{\mathrm{d}t}
        = \text{eom}(t, \text{state}, \text{pulse}),
    \qquad \text{state}(0) = y_0,
    \qquad 0 \le t \le t_\text{max}.
    $$

    The solver works on the scaled state `state / eom.state_scale(y0)`, so
    the tolerances in `config` are relative to `R0` for the radius and to
    $\sqrt{P_\text{amb}/\rho_L}$ for the wall velocity, and returns the
    solution in SI units. It evaluates the equation of motion only at
    admissible states (see
    [`EquationOfMotion.is_admissible`][jbubble.bubble.eom.EquationOfMotion.is_admissible]);
    elsewhere the right-hand side is NaN, so the adaptive solver rejects
    the trial step. Rejected steps never enter the solution, so the guard
    keeps gradients finite without changing the forward solve: with an
    explicit solver, such as the default `Dopri5`, the solution is the same,
    bit for bit, as without the guard. With an implicit solver, XLA can
    round the Newton iterations differently in the last bit, so the
    solution agrees to within the solver tolerance.

    The integration doesn't raise when the solver fails (`throw=False`);
    check `diffrax.is_successful` on the solution's `result`.

    Parameters
    ----------
    eom : EquationOfMotion
        Assembled equation of motion, such as
        [`KellerMiksis`][jbubble.bubble.eom.KellerMiksis].
    pulse : Pulse
        Driving acoustic pulse. An adaptive step-size controller steps to
        every time in
        [`pulse.window_edges`][jbubble.pulse.base.Pulse.window_edges], so
        it can't step over a pulse that starts late.
    y0 : BubbleState, optional
        Initial state in SI units. `None` uses `eom.initial_state()`. A zero
        `R0` or `P_gas0`, the `BubbleState` defaults, means "unset":
        `solve_eom` fills it from the equation of motion, so
        `BubbleState(R=1.2 * R0)` starts at rest at 1.2 times the equilibrium
        radius. [`EquationOfMotion.initial_state`][jbubble.bubble.eom.EquationOfMotion.initial_state]
        with `R=` and `R_dot=` builds the same state explicitly. For an
        empty cavity, set `P_gas0` to a tiny positive value, such as `1e-12`.
    t_max : float, optional
        Integration end time [s]. `None` uses `pulse.t_end`.
    save_spec : SaveSpec, optional
        Output sampling specification. `None` uses 1024 evenly spaced time
        points.
    config : SolverConfig, optional
        Numerical integration settings. `None` uses
        [`SolverConfig()`][jbubble.solver.SolverConfig].
    adjoint : diffrax.AbstractAdjoint, optional
        How `jax.grad` differentiates through the solve. `None` uses
        `diffrax.RecursiveCheckpointAdjoint()`, which gives the exact
        gradient of the discretised solution. Pass `diffrax.ForwardMode()`
        to use `jax.jacfwd`, for example in a Levenberg-Marquardt fit.
        diffrax advises against `diffrax.BacksolveAdjoint`, whose gradients
        are approximate.
    progress : bool
        Whether to show a text progress meter. Default: `False`.

    Returns
    -------
    diffrax.Solution
        Solution object with `ts`, and `ys` in SI units.
    """
    if save_spec is None:
        save_spec = SaveSpec()
    assert isinstance(save_spec, SaveSpec)

    if config is None:
        config = SolverConfig()
    assert isinstance(config, SolverConfig)

    y0 = eom.initial_state() if y0 is None else _with_equilibrium(eom, y0)
    scale = jax.lax.stop_gradient(eom.state_scale(y0))
    z0 = jtu.tree_map(jnp.divide, y0, scale)

    t0 = jnp.asarray(0.0)
    t1 = jnp.asarray(pulse.t_end if t_max is None else t_max)
    saveat = save_spec.build(t0, t1)

    progress_meter = (
        diffrax.TextProgressMeter() if progress else diffrax.NoProgressMeter()
    )
    _adjoint = adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint()

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_guarded_vector_field),
        config.solver,
        t0=t0,
        t1=t1,
        dt0=config.dt0,
        y0=z0,
        args=(eom, pulse, jax.lax.stop_gradient(z0), scale),
        saveat=saveat,
        stepsize_controller=_step_to_window_edges(config.stepsize_controller, pulse),
        max_steps=config.max_steps,
        progress_meter=progress_meter,
        throw=False,
        adjoint=_adjoint,
    )
    ys = jtu.tree_map(jnp.multiply, sol.ys, scale)
    return eqx.tree_at(lambda s: s.ys, sol, ys)
