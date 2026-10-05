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

    where $h = h_\text{frac} R_0$ is the van der Waals hard-core radius.

    Parameters
    ----------
    gamma : float or Property
        Polytropic exponent.
    h_frac : float or Property
        Hard-core radius as a fraction of `R0` (dimensionless). Example 07
        (`examples/07_cavitation_regimes.py`) uses `1 / 5.61` ≈ 0.178.
    """

    gamma: Property = eqx.field(converter=as_property)
    h_frac: Property = eqx.field(converter=as_property)

    def __call__(self, state: BubbleState) -> jax.Array:
        h = self.h_frac(state) * state.R0
        return state.P_gas0 * (
            (state.R0**3 - h**3) / (state.R**3 - h**3)
        ) ** self.gamma(state)
