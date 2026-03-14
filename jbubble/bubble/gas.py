"""Gas pressure models.

Computes the outward gas pressure p_gas as a function of the
instantaneous bubble state.  All models read ``state.R0`` and
``state.P_gas0`` directly from the state, so they need no separate
storage of equilibrium parameters.  The ``EquationOfMotion`` seeds
these fields in ``initial_state()`` and the ODE carries them forward
(frozen at zero derivative in the standard case).
"""

import abc

import equinox as eqx
import jax

from jbubble.bubble.properties import Property, as_property

from .state import BubbleState


class GasModel(eqx.Module, abc.ABC):
    """Internal gas pressure model.

    Computes the outward gas pressure p_gas as a function of the
    instantaneous state.  The equilibrium configuration (``R0``,
    ``P_gas0``) is read directly from ``state``, so gas models only
    store their intrinsic physical parameters (e.g. the polytropic
    exponent).

    Examples: polytropic law, van der Waals corrected gas.
    """

    @abc.abstractmethod
    def __call__(self, state: BubbleState) -> jax.Array:
        """Compute gas pressure p_gas(state).

        Parameters
        ----------
        state : BubbleState
            Current bubble state.  Uses ``state.R``, ``state.R0``,
            and ``state.P_gas0``.

        Returns
        -------
        scalar
            Gas pressure.
        """
        ...


class PolytropicGas(GasModel):
    """Polytropic gas law.

    p_gas(R) = P_gas0 (R0 / R)^(3 gamma)

    Fields
    ------
    gamma : float
        Polytropic exponent  (1.0 = isothermal, 1.4 = adiabatic air).
    """

    gamma: Property = eqx.field(converter=as_property)

    def __call__(self, state: BubbleState) -> jax.Array:
        return state.P_gas0 * (state.R0 / state.R) ** (3.0 * self.gamma(state))


class VanDerWaalsGas(GasModel):
    """Hard-core corrected polytropic gas (van der Waals).

    p_gas(R) = P_gas0 ((R0^3 - h^3) / (R^3 - h^3))^gamma

    where *h = h_frac * R0* is the van der Waals hard-core radius.

    Fields
    ------
    gamma : float
        Polytropic exponent.
    h_frac : float
        Hard-core radius as a fraction of R0  (dimensionless).
        A common value for lipid shells is 1/5.61 ≈ 0.178.
    """

    gamma: Property = eqx.field(converter=as_property)
    h_frac: Property = eqx.field(converter=as_property)

    def __call__(self, state: BubbleState) -> jax.Array:
        h = self.h_frac(state) * state.R0
        return (
            state.P_gas0
            * ((state.R0**3 - h**3) / (state.R**3 - h**3)) ** self.gamma(state)
        )
