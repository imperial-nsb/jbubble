r"""Gas pressure models.

Each model computes the outward gas pressure $p_\text{gas}$ as a function
of the instantaneous bubble state. All models read `state.R0` and
`state.P_gas0` directly from the state, so they need no separate storage
of equilibrium parameters. The
[`EquationOfMotion`][jbubble.bubble.eom.EquationOfMotion] seeds these
fields in
[`initial_state`][jbubble.bubble.eom.EquationOfMotion.initial_state], and
the ODE carries them forward, frozen at zero derivative in the standard
case.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax

from .property import Property, as_property
from .state import BubbleState

__all__ = ["GasModel", "PolytropicGas", "VanDerWaalsGas"]


class GasModel(eqx.Module, abc.ABC):
    r"""Internal gas pressure model.

    Computes the outward gas pressure $p_\text{gas}$ as a function of the
    instantaneous state. The model reads the equilibrium configuration,
    `R0` and `P_gas0`, directly from `state`, so gas models store only
    their intrinsic physical parameters, such as the polytropic exponent.

    Concrete models include the polytropic law
    ([`PolytropicGas`][jbubble.bubble.gas.PolytropicGas]) and the
    van der Waals corrected gas
    ([`VanDerWaalsGas`][jbubble.bubble.gas.VanDerWaalsGas]).
    """

    @abc.abstractmethod
    def __call__(self, state: BubbleState) -> jax.Array:
        r"""Compute the gas pressure, $p_\text{gas}(\text{state})$.

        Parameters
        ----------
        state : BubbleState
            Current bubble state. The model uses `state.R`, `state.R0`,
            and `state.P_gas0`.

        Returns
        -------
        jax.Array
            Scalar gas pressure [Pa].
        """
        ...

    def is_admissible(self, state: BubbleState) -> jax.Array:
        """Return whether the gas law is defined at `state`.

        The default requires a positive radius, `R > 0`. A gas law with a
        smaller domain, such as
        [`VanDerWaalsGas`][jbubble.bubble.gas.VanDerWaalsGas], overrides
        this method. [`solve_eom`][jbubble.solver.solve_eom] uses it,
        through
        [`EquationOfMotion.is_admissible`][jbubble.bubble.eom.EquationOfMotion.is_admissible],
        to keep gradients finite when a rejected trial step leaves the
        domain.

        Parameters
        ----------
        state : BubbleState
            State to check. Its leaves may be NaN or infinite.

        Returns
        -------
        jax.Array
            Boolean scalar.
        """
        return state.R > 0


class PolytropicGas(GasModel):
    r"""Polytropic gas law.

    $$
    p_\text{gas}(R) = P_{\text{gas},0}\left(\frac{R_0}{R}\right)^{3\gamma}
    $$

    Parameters
    ----------
    gamma : float or Property
        Polytropic exponent: `1.0` is isothermal and `1.4` is adiabatic air.
    """

    gamma: Property = eqx.field(converter=as_property)

    def __call__(self, state: BubbleState) -> jax.Array:
        return state.P_gas0 * (state.R0 / state.R) ** (3.0 * self.gamma(state))


class VanDerWaalsGas(GasModel):
    r"""Hard-core corrected polytropic gas (van der Waals).

    $$
    p_\text{gas}(R) = P_{\text{gas},0}
        \left(\frac{R_0^3 - h^3}{R^3 - h^3}\right)^{\gamma}
    $$

    where $h = h_\text{frac} R_0$ is the van der Waals hard-core radius: the
    radius of the bubble's gas content compressed to its excluded volume. The
    hard core is a property of the gas, not of the bubble, so `h_frac`
    depends on the gas species and its equilibrium density. The pressure
    diverges as $R \to h$ and the law is undefined for $R \le h$, which
    [`is_admissible`][jbubble.bubble.gas.VanDerWaalsGas.is_admissible]
    reports to the solver.

    Parameters
    ----------
    gamma : float or Property
        Polytropic exponent.
    h_frac : float or Property
        Hard-core radius as a fraction of `R0` (dimensionless). Example 04
        (`examples/04_equations_of_motion.py`) uses `1 / 8.86` ≈ 0.113.
    """

    gamma: Property = eqx.field(converter=as_property)
    h_frac: Property = eqx.field(converter=as_property)

    def __call__(self, state: BubbleState) -> jax.Array:
        h = self.h_frac(state) * state.R0
        return state.P_gas0 * (
            (state.R0**3 - h**3) / (state.R**3 - h**3)
        ) ** self.gamma(state)

    def is_admissible(self, state: BubbleState) -> jax.Array:
        """Return whether `R` lies outside the hard core, `R > h_frac * R0`.

        Parameters
        ----------
        state : BubbleState
            State to check. Its leaves may be NaN or infinite.

        Returns
        -------
        jax.Array
            Boolean scalar.
        """
        hard_core = self.h_frac(state) * state.R0
        return hard_core < state.R
