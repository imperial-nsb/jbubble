"""Unit-scaling helpers for jbubble simulations."""

import equinox as eqx


class Units(eqx.Module):
    """Non-dimensionalisation factors shared across the library.

    The defaults normalise micrometre radii and microsecond times, which keeps
    the Rayleigh-Plesset solver numerically well-conditioned while remaining
    easy to interpret when converting back to SI units later on.
    """

    L_scale: float = 1e-6
    T_scale: float = 1e-6
    M_scale: float = 1e-15

    @property
    def rho_scale(self) -> float:
        return self.M_scale / self.L_scale**3

    @property
    def P_scale(self) -> float:
        return self.M_scale / (self.L_scale * self.T_scale**2)

    @property
    def vel_scale(self) -> float:
        return self.L_scale / self.T_scale

    @property
    def force_scale(self) -> float:
        return self.M_scale * self.L_scale / self.T_scale**2

    @property
    def sigma_scale(self) -> float:
        return self.M_scale / self.T_scale**2

    @property
    def chi_scale(self) -> float:
        return self.sigma_scale

    @property
    def mu_scale(self) -> float:
        return self.M_scale / (self.L_scale * self.T_scale)

    @property
    def kappa_scale(self) -> float:
        return self.M_scale / self.T_scale

    @property
    def freq_scale(self) -> float:
        return 1.0 / self.T_scale
