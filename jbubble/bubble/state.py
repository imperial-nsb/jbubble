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
