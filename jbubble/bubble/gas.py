"""Gas pressure models.

Computes the outward gas pressure p_gas as a function of the
instantaneous bubble state.
"""

import abc

import equinox as eqx
import jax

from .state import BubbleState


class GasModel(eqx.Module, abc.ABC):
    """Internal gas pressure model.

    Computes the outward gas pressure p_gas as a function of the
    instantaneous radius R only.  The gas state is determined by bubble
    volume, so it has no dependence on wall velocity Rdot.

    Examples: polytropic law, van der Waals corrected gas.
    """

    @abc.abstractmethod
    def __call__(self, state: BubbleState) -> jax.Array:
        """Compute gas pressure p_gas(R).

        Parameters
        ----------
        state : BubbleState
            Current bubble state.

        Returns
        -------
        scalar
            Gas pressure p_gas(R).
        """
        ...


class PolytropicGas(GasModel):
    """Polytropic gas law.

    p_gas(R) = P_gas0 (R0 / R)^(3 gamma)

    Fields
    ------
    P_gas0 : float
        Equilibrium gas pressure  [Pa].
    R0 : float
        Equilibrium bubble radius  [m].
    gamma : float
        Polytropic exponent  (1.0 = isothermal, 1.4 = adiabatic air).
    """

    P_gas0: float
    R0: float
    gamma: float


    def __call__(self, state: BubbleState) -> jax.Array:
        return self.P_gas0 * (self.R0 / state.R) ** (3.0 * self.gamma)


class VanDerWaalsGas(GasModel):
    """Hard-core corrected polytropic gas (van der Waals).

    p_gas(R) = P_gas0 ((R0^3 - h^3) / (R^3 - h^3))^gamma

    where *h* is the van der Waals hard-core radius.

    Fields
    ------
    P_gas0 : float
        Equilibrium gas pressure  [Pa].
    R0 : float
        Equilibrium bubble radius  [m].
    gamma : float
        Polytropic exponent.
    h : float
        Van der Waals hard-core radius  [m].
    """

    P_gas0: float
    R0: float
    gamma: float
    h: float


    def __call__(self, state: BubbleState) -> jax.Array:
        R = state.R
        return (
            self.P_gas0 * ((self.R0**3 - self.h**3) / (R**3 - self.h**3)) ** self.gamma
        )
