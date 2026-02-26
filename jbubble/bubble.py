"""
Multibubble module

Bubble material model definitions for single-bubble dynamics.
Includes Rayleigh-Plesset, Marmottant, Gompertz-smoothed shell,
and Kelvin-Voigt viscoelastic models.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from .units import Units
from .pulse import Pulse
from typing import Any, Tuple

State = jax.Array


class Bubble(eqx.Module):
    """
    Abstract base class for bubble material models.

    All bubble models must implement:
        - surface_tension(R)
        - chi_R(R)
        - get_scaled(units)
        - bubble_equation(t, state, pulse)
        - initial_state()

    These functions define the mechanical response of the interface.
    """

    def surface_tension(self, R: jax.Array) -> jax.Array:
        raise NotImplementedError

    def chi_R(self, R: jax.Array) -> jax.Array:
        raise NotImplementedError
    
    def get_scaled(self, units: Units) -> "BubbleBase":
        raise NotImplementedError

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        raise NotImplementedError
    
    def initial_state(self) -> jax.Array:
        return jnp.array([self.R0, 0.0])

    def rescale_state(self, state: jax.Array, units: Units) -> jax.Array:
        """Rescale the state variables back to physical units.
        Override this for models with different state definitions.
        """
        scale_factors = jnp.array([units.L_scale, units.vel_scale])
        return state * scale_factors


class RayleighPlesset(Bubble):
    """
    Rayleigh-Plesset model for an uncoated gas bubble
    in an incompressible Newtonian liquid.

    Assumptions:
        - Constant surface tension
        - Polytropic gas behavior
        - Incompressible surrounding liquid
    """

    R0: jax.Array
    gamma: float
    mu_L: float
    rho_L: float
    c_L: float
    P_amb: float
    sigma_L: float
    sigma_R0: float

    def __init__(
        self,
        R0: float,
        gamma: float = 1.4,
        mu_L: float = 0.00089,
        rho_L: float = 1000.0,
        c_L: float = 1498.0,
        P_amb: float = 101.3e3,
        sigma_L: float = 72e-3,
    ) -> None:

        self.R0 = R0
        self.gamma = gamma
        self.mu_L = mu_L
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L

        self.sigma_R0 = self.sigma_L

    def surface_tension(self, R: jax.Array) -> jax.Array:
        return self.sigma_L * jnp.ones_like(R)

    def chi_R(self, R: jax.Array) -> jax.Array:
        return jnp.zeros_like(R)

    def get_scaled(self, units: Units) -> "RayleighPlesset":
        return RayleighPlesset(
            R0=self.R0 / units.L_scale,
            gamma=self.gamma,
            mu_L=self.mu_L / units.mu_scale,
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale,
        )

    def bubble_equation(self, t: Any, state: State, pulse) -> State:

        R, R_dot= state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0
        P_gas = P_gas0 * ((self.R0**3) / (R**3)) ** self.gamma

        P_surf = 2.0 * sigma / R
        P_visc = 4.0 * self.mu_L * R_dot / R

        forces = P_gas - P_surf - P_visc - P_drive - self.P_amb
        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])


class Marmottant(Bubble):
    """
    Marmottant shell model for encapsulated microbubbles.
    A lipid-coated microbubble in a Newtonian fluid following a discontinuous piecewise surface tension law.

    Surface tension follows three regimes:
        1) Buckled:     σ = 0
        2) Elastic:     σ = χ (R²/R_buckle² − 1)
        3) Ruptured:    σ = σ_L

    Includes:
        - Shell elasticity and shell viscosity
        - Van der Waals gas correction
        - Liquid compressibility correction
    """

    R0: jax.Array
    R_buckle: float
    gamma: float
    chi: float
    mu_L: float
    kappa_s: float
    rho_L: float
    c_L: float
    P_amb: float
    sigma_L: float

    R_break: float
    sigma_break: float
    sigma_R0: float
    vdw: float

    def __init__(
        self,
        R0: float,
        R_buckle: float | None = None,
        gamma: float = 1.07,
        chi: float = 0.38,
        mu_L: float = 0.00089,
        kappa_s: float = 2.4e-9,
        rho_L: float = 1000.0,
        c_L: float = 1498.0,
        P_amb: float = 101.3e3,
        sigma_L: float = 72e-3,
        vdw_divisor: float = 5.61,
    ) -> None:

        self.R0 = R0
        self.R_buckle = 0.99 * self.R0 if R_buckle is None else R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_L = mu_L
        self.kappa_s = kappa_s
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L

        self.sigma_break = self.sigma_L
        self.R_break = jnp.sqrt((self.sigma_break/self.chi) + 1) * self.R_buckle
        self.sigma_R0 = self.chi * ((self.R0**2 / self.R_buckle**2) - 1.0)
        self.vdw = self.R0 / vdw_divisor

    def surface_tension(self, R: jax.Array) -> jax.Array:

        sigma_elastic = self.chi * ((R**2 / self.R_buckle**2) - 1.0)

        return jnp.where(
            R <= self.R_buckle,
            0.0,
            jnp.where(R >= self.R_break, self.sigma_L, sigma_elastic),
        )

    def chi_R(self, R: jax.Array) -> jax.Array:
        d_sigma = jax.grad(lambda r: self.surface_tension(r))
        d_sigma_dR = jax.vmap(d_sigma)(R)
        return 0.5 * R * d_sigma_dR
    
    def get_scaled(self, units: Units) -> "Marmottant":
        return Marmottant(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,
            gamma=self.gamma,
            chi=self.chi * units.sigma_scale,
            mu_L=self.mu_L / units.mu_scale,
            kappa_s=self.kappa_s / (units.mu_scale * units.L_scale),
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale,
            vdw_divisor=5.61,  # Keep vdw scaling consistent with R0
        )

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:

        R, R_dot= state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0
        P_gas = P_gas0 * ((self.R0**3 - self.vdw**3) /
                         (R**3 - self.vdw**3)) ** self.gamma

        P_surf = 2.0 * sigma / R
        P_visc = 4.0 * self.mu_L * R_dot / R
        P_surf_visc = 4.0 * self.kappa_s * R_dot / R**2

        damping_term = 1.0 - (
            3.0 * self.gamma * R_dot * R**3
        ) / (self.c_L * (R**3 - self.vdw**3))

        forces = (P_gas * damping_term
                  - P_surf - P_visc - P_surf_visc
                  - P_drive - self.P_amb)

        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])


class MarmottantGompertz(Bubble):
    """
    Smooth Gompertz-based variant of the Marmottant shell model.

    The discontinuous piecewise surface tension law is replaced with
    a differentiable Gompertz function, making the model suitable for:

        - Automatic differentiation
        - Gradient-based optimization
        - Inverse problems
        - Machine learning applications

    Includes:
        - Shell elasticity and shell viscosity
        - Van der Waals gas correction
        - Liquid compressibility correction
    """

    R0: jax.Array
    R_buckle: float
    gamma: float
    chi: jax.Array
    mu_L: float
    kappa_s: jax.Array
    rho_L: float
    c_L: float
    P_amb: float
    sigma_L: float

    R_break: float
    sigma_break: float
    sigma_R0: float
    vdw: float

    def __init__(
        self,
        R0: float,
        R_buckle: float | None = None,
        gamma: float = 1.07,
        chi: float = 0.38,
        mu_L: float = 0.00089,
        kappa_s: float = 2.4e-9,
        rho_L: float = 1000.0,
        c_L: float = 1498.0,
        P_amb: float = 101.3e3,
        sigma_L: float = 72e-3,
        vdw_divisor: float = 5.61,
    ) -> None:

        self.R0 = R0
        self.R_buckle = 0.99 * self.R0 if R_buckle is None else R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_L = mu_L
        self.kappa_s = kappa_s
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L

        self.R_break = 1.1 * self.R0
        self.sigma_break = self.sigma_L
        self.sigma_R0 = self.chi * ((self.R0**2 / self.R_buckle**2) - 1.0)
        self.vdw = self.R0 / vdw_divisor

    def surface_tension(self, R: jax.Array) -> jax.Array:
        """
        Gompertz surface tension law.

        Provides a smooth transition between buckled,
        elastic, and ruptured regimes.
        """

        a = self.sigma_break
        c = (2 * self.chi / a) * jnp.sqrt(1 + a / (2 * self.chi))
        b = -jnp.log(self.sigma_R0 / a) / jnp.exp(
            c * (1 - self.R0 / self.R_buckle)
        )

        sigma = a * jnp.exp(-b * jnp.exp(c * (1 - R / self.R_buckle)))

        return sigma

    def chi_R(self, R: jax.Array) -> jax.Array:
        """
        χ(R) = (R/2) dσ/dR

        Computed via automatic differentiation to preserve consistency
        if the surface tension law is modified.
        """

        d_sigma = jax.grad(lambda r: self.surface_tension(r))
        d_sigma_dR = jax.vmap(d_sigma)(R)
        return 0.5 * R * d_sigma_dR
    
    def get_scaled(self, units: Units) -> "MarmottantGompertz":
        return MarmottantGompertz(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,
            gamma=self.gamma,
            chi=self.chi * units.sigma_scale,
            mu_L=self.mu_L / units.mu_scale,
            kappa_s=self.kappa_s / (units.mu_scale * units.L_scale),
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale,
            vdw_divisor=5.61,  # Keep vdw scaling consistent with R0
        )

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:

        R, R_dot = state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0
        P_gas = P_gas0 * (
            (self.R0**3 - self.vdw**3) /
            (R**3 - self.vdw**3)
        ) ** self.gamma

        P_surf = 2.0 * sigma / R
        P_visc = 4.0 * self.mu_L * R_dot / R
        P_surf_visc = 4.0 * self.kappa_s * R_dot / R**2

        damping_term = 1.0 - (
            3.0 * self.gamma * R_dot * R**3
        ) / (self.c_L * (R**3 - self.vdw**3))

        forces = (
            P_gas * damping_term
            - P_surf
            - P_visc
            - P_surf_visc
            - P_drive
            - self.P_amb
        )

        R_ddot = (forces / self.rho_L - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])


class KelvinVoigtGompertz(Bubble):
    """
    Lipid-coated microbubble in a Kelvin-Voigt viscoelastic medium with
    Gompertz shell surface tension.

    The surrounding material is modeled as a Kelvin-Voigt solid:
        σ_medium = G * strain + μ_m * strain_rate

    Includes:
        - Medium viscosity (μ_m)
        - Medium elastic modulus (G)
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law

    No liquid compressibility correction is included.
    """

    R0: jax.Array
    R_buckle: float
    gamma: float
    chi: float
    mu_m: float
    kappa_s: float
    rho_m: float
    P_amb: float
    sigma_m: float
    G: float

    R_break: float
    sigma_break: float
    sigma_R0: float

    def __init__(
        self,
        R0: float,
        R_buckle: float | None = None,
        gamma: float = 1.07,
        chi: float = 0.38,
        mu_m: float = 0.00089,
        kappa_s: float = 2.4e-9,
        rho_m: float = 1000.0,
        P_amb: float = 101.3e3,
        sigma_m: float = 72e-3,
        G: float = 0.0,
    ) -> None:

        self.R0 = R0
        self.R_buckle = 0.99 * self.R0 if R_buckle is None else R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_m = mu_m
        self.kappa_s = kappa_s
        self.rho_m = rho_m
        self.P_amb = P_amb
        self.sigma_m = sigma_m
        self.G = G

        self.R_break = 1.1 * self.R0
        self.sigma_break = self.sigma_m
        self.sigma_R0 = self.chi * ((self.R0**2 / self.R_buckle**2) - 1.0)

    def surface_tension(self, R: jax.Array) -> jax.Array:
        """
        Gompertz surface tension law.
        """

        a = self.sigma_break
        c = (2 * self.chi / a) * jnp.sqrt(1 + a / (2 * self.chi))
        b = -jnp.log(self.sigma_R0 / a) / jnp.exp(
            c * (1 - self.R0 / self.R_buckle)
        )

        return a * jnp.exp(-b * jnp.exp(c * (1 - R / self.R_buckle)))

    def chi_R(self, R: jax.Array) -> jax.Array:

        d_sigma = jax.grad(lambda r: self.surface_tension(r))
        d_sigma_dR = jax.vmap(d_sigma)(R)
        return 0.5 * R * d_sigma_dR

    def get_scaled(self, units: Units) -> "KelvinVoigtGompertz":
        return KelvinVoigtGompertz(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,
            gamma=self.gamma,
            chi=self.chi * units.sigma_scale,
            mu_m=self.mu_m / units.mu_scale,
            kappa_s=self.kappa_s / (units.mu_scale * units.L_scale),
            rho_m=self.rho_m / units.rho_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_m=self.sigma_m / units.sigma_scale,
            G=self.G / units.P_scale,
        )
    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:

        R, R_dot= state

        sigma = self.surface_tension(R)
        P_drive = pulse(t)

        P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0
        P_gas = P_gas0 * (self.R0 / R) ** (3 * self.gamma)

        P_surf = 2.0 * sigma / R
        P_visc = 4.0 * self.mu_m * R_dot / R
        P_surf_visc = 4.0 * self.kappa_s * R_dot / R**2

        P_elastic = (4.0 / 3.0) * self.G * (
            (R**3 - self.R0**3) / self.R0**3
        )

        forces = (
            P_gas
            - P_surf
            - P_visc
            - P_surf_visc
            - P_drive
            - self.P_amb
            - P_elastic
        )

        R_ddot = (forces / self.rho_m - 1.5 * R_dot**2) / R

        return jnp.stack([R_dot, R_ddot])


class KellerMiksisGompertz(Bubble):
    """
    Keller-Miksis-Gompertz Model:
    A lipid-coated microbubble in a non-newtonian fluid following a differentiable surface tension law (Gompertz)

    Includes:
        - Shell elasticity and viscosity (chi and kappa_s)
        - Smooth Gompertz surface tension
        - Polytropic gas law
        - Liquid compressibility correction based on Keller-Miksis formulation

    """

    R0: jax.Array
    R_buckle: float
    gamma: float
    chi: float
    mu_L: float
    kappa_s: float
    rho_L: float
    c_L: float
    P_amb: float
    sigma_L: float

    R_break: float
    sigma_break: float
    sigma_R0: float
    P_gas0: float

    def __init__(
        self,
        R0: float,
        R_buckle: float | None = None,
        gamma: float = 1.07,
        chi: float = 0.38,
        mu_L: float = 0.00089,
        kappa_s: float = 2.4e-9,
        rho_L: float = 1000.0,
        c_L: float = 1498.0,
        P_amb: float = 101.3e3,
        sigma_L: float = 72e-3,
    ) -> None:
        self.R0 = R0
        self.R_buckle = 0.99 * self.R0 if R_buckle is None else R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_L = mu_L
        self.kappa_s = kappa_s
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L

        # Shell rupture
        self.R_break = 1.1 * R0
        self.sigma_break = sigma_L
        self.sigma_R0 = chi * ((R0**2 / self.R_buckle**2) - 1.0)

        # Precompute gas pressure at equilibrium
        self.P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0

    def surface_tension(self, R: jax.Array) -> jax.Array:
        a = self.sigma_break
        c = (2 * self.chi / a) * jnp.sqrt(1 + a / (2 * self.chi))
        b = -jnp.log(self.sigma_R0 / a) / jnp.exp(c * (1 - self.R0 / self.R_buckle))
        return a * jnp.exp(-b * jnp.exp(c * (1 - R / self.R_buckle)))

    def get_scaled(self, units: Units) -> "KellerMiksisGompertz":
        return KellerMiksisGompertz(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,
            gamma=self.gamma,
            chi=self.chi * units.sigma_scale,
            mu_L=self.mu_L / units.mu_scale,
            kappa_s=self.kappa_s / (units.mu_scale * units.L_scale),
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale,
        )

    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        R, R_dot = state

        sigma = self.surface_tension(R)
        dsigma_dR = jax.grad(lambda r: self.surface_tension(r))(R)
        P_drive = pulse(t)

        # Pressure components
        P_Laplace = 2.0 * sigma / R
        P_visc = 4.0 * self.mu_L * R_dot / R
        P_surf_visc = 4.0 * self.kappa_s * R_dot / R**2

        dPg_dt = (-3.0 * self.gamma) * self.P_gas0 * (R_dot / R) * (self.R0 / R) ** (3 * self.gamma)
        dPdrive_dt = jax.grad(lambda t_val: pulse(t_val))(t)
        dLaplace_dt = 2 * R_dot * (dsigma_dR / R - sigma / R**2)

        P_int = self.P_gas0 * (self.R0 / R) ** (3 * self.gamma)

        A = (1 / self.rho_L) * ((1 + R_dot / self.c_L) * (P_int - P_drive - self.P_amb - P_Laplace - P_visc - P_surf_visc))
        B = dPg_dt - dPdrive_dt - dLaplace_dt
        C = R * (1 - R_dot / self.c_L) + (1 / self.rho_L) * (R / self.c_L) * (4 * self.kappa_s / R**2 + 4 * self.mu_L / R)

        D = A + (1 / self.rho_L) * (R / self.c_L) * (B + (8 * self.kappa_s * R_dot**2 / R**3) + (4 * self.mu_L * R_dot**2 / R**2)) \
            - 1.5 * (1 - R_dot / (3 * self.c_L)) * R_dot**2

        R_ddot = D / C
        return jnp.stack([R_dot, R_ddot])


class LeightonGompertz(Bubble):
    """
    Leighton Model:
    A lipid-coated microbubble confined in a rigid-walled tube and following a differentiable surface tension law (Gompertz)

    Models a bubble confined within a rigid-walled tube, incorporating:
    - Shell elasticity and viscosity (chi and kappa_s)
    - Smooth Gompertz surface tension
    - Polytropic gas law
    - Tube wall inertia and geometry effects
    - Liquid compressibility correction
    """

    R0: jax.Array
    R_buckle: float
    gamma: float
    chi: float
    mu_L: float
    kappa_s: float
    rho_L: float
    c_L: float
    P_amb: float
    sigma_L: float
    tube_radius: float
    tube_length: float

    R_break: float
    sigma_break: float
    sigma_R0: float
    P_gas0: float

    def __init__(
        self,
        R0: float,
        R_buckle: float | None = None,
        gamma: float = 1.07,
        chi: float = 0.38,
        mu_L: float = 0.00089,
        kappa_s: float = 2.4e-9,
        rho_L: float = 1000.0,
        c_L: float = 1498.0,
        P_amb: float = 101.3e3,
        sigma_L: float = 72e-3,
        tube_radius: float = 10.0e-6,
        tube_length = 100.0e-6,
    ) -> None:

        self.R0 = R0
        self.R_buckle = 0.99 * self.R0 if R_buckle is None else R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_L = mu_L
        self.kappa_s = kappa_s
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L
        self.tube_radius = tube_radius
        self.tube_length = tube_length

        # Shell rupture
        self.R_break = 1.1 * R0
        self.sigma_break = sigma_L
        self.sigma_R0 = chi * ((R0**2 / self.R_buckle**2) - 1.0)

        # Precompute gas pressure at equilibrium
        self.P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0

    def surface_tension(self, R: jax.Array) -> jax.Array:

        a = self.sigma_break
        c = (2 * self.chi / a) * jnp.sqrt(1 + a / (2 * self.chi))
        b = -jnp.log(self.sigma_R0 / a) / jnp.exp(
            c * (1 - self.R0 / self.R_buckle)
        )

        return a * jnp.exp(-b * jnp.exp(c * (1 - R / self.R_buckle)))

    def get_scaled(self, units: Units) -> "LeightonGompertz":  
        return LeightonGompertz(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,
            gamma=self.gamma,
            chi=self.chi * units.sigma_scale,
            mu_L=self.mu_L / units.mu_scale,
            kappa_s=self.kappa_s / (units.mu_scale * units.L_scale),
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale,
            tube_radius = self.tube_radius / units.L_scale,
            tube_length = self.tube_length / units.L_scale,
        )
    
    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        R, R_dot = state

        sigma = self.surface_tension(R)

        tuberad = self.tube_radius     # Γ1 (tube radius)
        zeta = self.tube_length / 2.0  # ζ1 (half-length)

        # Geometry factor per Leighton:
        # alpha = (ζ1/Γ1) * (1 + 8Γ1/(3πζ1)) - 1
        alpha = (zeta / tuberad) * (1.0 + (8.0 * tuberad) / (3.0 * jnp.pi * zeta)) - 1.0
        beta  = 2.0 * alpha  # curly-brace factor in the paper

        # Driving & gas term with small-order compressibility
        P_drive = pulse(t)
        damping_term = 1.0 - (3.0 * self.gamma * R_dot) / self.c_L
        P_gas = self.P_gas0 * (self.R0 / R) ** (3.0 * self.gamma) * damping_term

        # Dissipative and capillary terms
        P_surface_visc = 4.0 * self.kappa_s * R_dot / (R**2)
        P_liq_visc     = 4.0 * self.mu_L * R_dot / R
        P_Laplace      = 2.0 * sigma / R

        # Net forcing (right-hand side)
        rhs = (P_gas - P_Laplace - P_liq_visc - P_surface_visc - P_drive - self.P_amb) / self.rho_L

        # Leighton tube-inertia multipliers:
        # R*R̈ * [1 + (R/Γ1) β] + (3/2) Ṙ² * [1 + (4R/(3Γ1)) β] = rhs
        denom = R * (1.0 + (R / tuberad) * beta)
        inert = 1.5 * (R_dot ** 2) * (1.0 + (4.0 * R) / (3.0 * tuberad) * beta)

        R_ddot = (rhs - inert) / denom
        return jnp.stack([R_dot, R_ddot])


class SphericalConfinement(Bubble):
    """
    Spherical confinement model with Gompertz surface tension law.

    Models a lipid-coated bubble confined within an thin, elastic and spherical shell.

    This model captures the key qualitative effects of confinement on bubble dynamics, such as the presence of two normal modes, while maintaining computational tractability.
    Includes:
    - Shell elasticity and viscosity (chi and kappa_s)
    - Smooth Gompertz surface tension
    - Polytropic gas law
    - Confinement force based on linear elasticity of the shell
    - Liquid compressibility correction.

    """

    R0: jax.Array
    R_buckle: float
    gamma: float
    chi: float
    mu_L: float
    kappa_s: float
    rho_L: float
    c_L: float
    P_amb: float
    sigma_L: float
    vessel_radius: float
    vessel_rho: float
    vessel_E: float
    vessel_d: float
    tissue_rho: float
    tissue_d: float

    R_break: float
    sigma_break: float
    sigma_R0: float
    P_gas0: float


    def __init__(
        self,
        R0: float,
        R_buckle: float | None = None,
        gamma: float = 1.07,
        chi: float = 0.38,
        mu_L: float = 0.00089,
        kappa_s: float = 2.4e-9,
        rho_L: float = 1000.0,
        c_L: float = 1498.0,
        P_amb: float = 101.3e3,
        sigma_L: float = 72e-3,
        vessel_radius: float = 15.0e-6,  
        vessel_rho: float = 900.0,  
        vessel_E: float = 1.0e6,
        vessel_d: float = 1.0e-6,
        tissue_rho: float = 900.0,
        tissue_d: float = 1.0e-6,
    ) -> None:

        self.R0 = R0
        self.R_buckle = 0.99 * self.R0 if R_buckle is None else R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_L = mu_L
        self.kappa_s = kappa_s
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L
        self.vessel_radius = vessel_radius
        self.vessel_rho = vessel_rho
        self.vessel_E = vessel_E
        self.vessel_d = vessel_d
        self.tissue_rho = tissue_rho
        self.tissue_d = tissue_d
        # Shell rupture
        self.R_break = 1.1 * R0
        self.sigma_break = sigma_L
        self.sigma_R0 = chi * ((R0**2 / self.R_buckle**2) - 1.0)

        # Precompute gas pressure at equilibrium
        self.P_gas0 = self.P_amb + 2.0 * self.sigma_R0 / self.R0

    def surface_tension(self, R: jax.Array) -> jax.Array:

        a = self.sigma_break
        c = (2 * self.chi / a) * jnp.sqrt(1 + a / (2 * self.chi))
        b = -jnp.log(self.sigma_R0 / a) / jnp.exp(
            c * (1 - self.R0 / self.R_buckle)
        )

        return a * jnp.exp(-b * jnp.exp(c * (1 - R / self.R_buckle)))

    def get_scaled(self, units: Units) -> "SphericalConfinement":  
        return SphericalConfinement(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,          
            gamma=self.gamma,
            chi=self.chi * units.sigma_scale,
            mu_L=self.mu_L / units.mu_scale,
            kappa_s=self.kappa_s / (units.mu_scale * units.L_scale),
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale,
            vessel_radius = self.vessel_radius / units.L_scale,
            vessel_rho = self.vessel_rho / units.rho_scale,
            vessel_E = self.vessel_E / units.P_scale,
            vessel_d = self.vessel_d / units.L_scale,
            tissue_rho = self.tissue_rho / units.rho_scale,
            tissue_d = self.tissue_d / units.L_scale,
        )
    
    def bubble_equation(self, t: Any, state: jax.Array, pulse) -> jax.Array:
        R, R_dot, a, a_dot = state
        sigma = self.surface_tension(R)
        P_drive = pulse(t)
        # Confinement force (simplified linear elastic response)

        rhoL = self.rho_L
        mu   = self.mu_L
        ks   = self.kappa_s
        P0   = self.P_amb
        a0   = self.vessel_radius
        Ev   = self.vessel_E
        nu   = 0.5   # nearly incompressible wall
        
        P_gas = self.P_gas0 * (self.R0 / R) ** (3.0 * self.gamma) \
                * (1.0 - 3.0 * self.gamma * R_dot / self.c_L)

        A = R**2
        B = -a**2
        C = rhoL * R**2 * (1.0/R - 1.0/a)
        D = self.vessel_rho * self.vessel_d + self.tissue_rho * self.tissue_d

        E = 2.0 * a * a_dot**2 - 2.0 * R * R_dot**2

        P_wall = Ev * (a - a0) / ((1.0 - nu**2) * a**2)

        F = (
            P_gas
            - 2.0 * R * R_dot * rhoL * (1.0/R - 1.0/a)
            - 2.0 * sigma / R
            - 4.0 * mu * (R_dot/R + a_dot/a)
            - 4.0 * ks * R_dot / R**2
            - P_wall
            - P0
            - P_drive
        )

        Delta = A * D - B * C

        Delta = jnp.where(jnp.abs(Delta) < 1e-14, 1e-14, Delta)

        R_ddot = (E * D - B * F) / Delta
        a_ddot = (A * F - C * E) / Delta
    
        return jnp.stack([R_dot, R_ddot, a_dot, a_ddot])
    
    def initial_state(self) -> jax.Array:   
        return jnp.array([self.R0, 0.0, self.vessel_radius, 0.0])

    def rescale_state(self, state: jax.Array, units: Units) -> jax.Array:
        scale_factors = jnp.array([
            units.L_scale, units.vel_scale, units.L_scale, units.vel_scale
        ])
        return state * scale_factors
