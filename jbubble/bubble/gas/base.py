
import abc

import equinox as eqx
import jax


class GasModel(eqx.Module, abc.ABC):
    """Internal gas pressure model.

    Computes the outward gas pressure p_gas as a function of the
    instantaneous radius R only.  The gas state is determined by bubble
    volume, so it has no dependence on wall velocity Rdot.

    Examples: polytropic law, van der Waals corrected gas.
    """

    @abc.abstractmethod
    def __call__(self, R: jax.Array) -> jax.Array:
        """Compute gas pressure p_gas(R).

        Parameters
        ----------
        R : scalar
            Current bubble radius.

        Returns
        -------
        scalar
            Gas pressure p_gas(R).
        """
        ...
