import jax

from . import _defaults
from .interfaces import GasModel


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

    @classmethod
    def from_equilibrium(
        cls,
        *,
        R0: float,
        gamma: float,
        P_amb: float,
        sigma_R0: float,
    ) -> "PolytropicGas":
        """Construct from physical equilibrium conditions.

        Computes  P_gas0 = P_amb + 2 sigma(R0) / R0.

        Parameters
        ----------
        R0 : float
            Equilibrium radius  [m].
        gamma : float
            Polytropic exponent.
        P_amb : float
            Ambient pressure  [Pa].
        sigma_R0 : float
            Surface tension evaluated at R0  [N/m].
        """
        P_gas0 = P_amb + 2.0 * sigma_R0 / R0
        return cls(P_gas0=P_gas0, R0=R0, gamma=gamma)

    def __call__(self, R: jax.Array) -> jax.Array:
        return self.P_gas0 * (self.R0 / R) ** (3.0 * self.gamma)


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

    @classmethod
    def from_equilibrium(
        cls,
        *,
        R0: float,
        gamma: float,
        P_amb: float,
        sigma_R0: float,
        h_divisor: float = _defaults.VDW_DIVISOR,
    ) -> "VanDerWaalsGas":
        """Construct from equilibrium conditions.

        Computes  P_gas0 = P_amb + 2 sigma(R0) / R0  and  h = R0 / h_divisor.

        Parameters
        ----------
        R0 : float
            Equilibrium radius  [m].
        gamma : float
            Polytropic exponent.
        P_amb : float
            Ambient pressure  [Pa].
        sigma_R0 : float
            Surface tension evaluated at R0  [N/m].
        h_divisor : float
            Divisor for hard-core radius  (default 5.61).
        """
        P_gas0 = P_amb + 2.0 * sigma_R0 / R0
        return cls(P_gas0=P_gas0, R0=R0, gamma=gamma, h=R0 / h_divisor)

    def __call__(self, R: jax.Array) -> jax.Array:
        return (
            self.P_gas0 * ((self.R0**3 - self.h**3) / (R**3 - self.h**3)) ** self.gamma
        )
