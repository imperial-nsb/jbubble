"""Bubble state PyTrees for ODE integration.

[`BubbleState`][jbubble.bubble.state.BubbleState] is the standard state
vector for a single bubble.

Equinox modules as ODE states guarantee strict PyTree congruency with
diffrax. To extend the physics, for example with thermal dynamics or
rectified diffusion, you add new fields.

The state carries the equilibrium fields `R0` and `P_gas0` alongside the
dynamic variables, so every gas and shell model reads them from the state
without separate storage or extra arguments. In the standard case their
time derivatives are zero (frozen constants).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["BubbleState"]


class BubbleState(eqx.Module):
    r"""Standard bubble state.

    Parameters
    ----------
    R : jax.Array
        Bubble wall radius [m].
    R_dot : jax.Array
        Bubble wall velocity [m/s]. Keyword-only. Default: `0`.
    R0 : jax.Array
        Equilibrium bubble radius [m]. Frozen
        ($\mathrm{d}R_0/\mathrm{d}t = 0$) in the standard case; it becomes
        a slow state variable for processes such as rectified diffusion.
        Keyword-only. Default: `0`.
    P_gas0 : jax.Array
        Equilibrium gas pressure [Pa]. Frozen in the standard case.
        Keyword-only. Default: `0`.

    Notes
    -----
    A zero `R0` or `P_gas0` means "unset":
    [`solve_eom`][jbubble.solver.solve_eom] fills it from the equation of
    motion, so `BubbleState(R=1.2 * R0)` starts at rest at 1.2 times the
    equilibrium radius.
    [`EquationOfMotion.initial_state`][jbubble.bubble.eom.EquationOfMotion.initial_state]
    builds the same state with `R=` and `R_dot=`. For an empty cavity, set
    `P_gas0` to a tiny positive value, such as `1e-12`, because zero means
    "unset".
    """

    R: jax.Array
    R_dot: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(()),
        kw_only=True,
    )
    R0: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(()),
        kw_only=True,
    )
    P_gas0: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(()),
        kw_only=True,
    )
