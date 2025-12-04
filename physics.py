import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable

class Units:
    def __init__(self, L_scale=1e-6, T_scale=1e-6, M_scale=1e-15):
        self.L_scale = L_scale
        self.T_scale = T_scale
        self.M_scale = M_scale

        # Derived scales
        self.rho_scale = M_scale / L_scale**3
        self.P_scale = M_scale / (L_scale * T_scale**2)
        self.vel_scale = L_scale / T_scale
        self.force_scale = M_scale * L_scale / T_scale**2
        self.sigma_scale = M_scale / T_scale**2  # N/m = kg/s^2
        self.chi_scale = self.sigma_scale        # N/m
        self.mu_scale = M_scale / (L_scale * T_scale) # Pa s = kg / (m s)
        self.kappa_scale = M_scale / T_scale     # N s / m = kg / s
        self.freq_scale = 1 / T_scale

class Bubble(eqx.Module):
    """
    Bubble physics parameters and Marmottant model logic.
    """
    R0: float  # Initial bubble radius (m)
    R_buckle: float  # Buckling radius (m)
    gamma: float  # Polytropic exponent (dimensionless)
    chi: float  # Shell elasticity / compression modulus (N/m)
    mu_L: float  # Liquid viscosity (Pa·s)
    kappa_s: float  # Shell viscosity (N·s/m)
    rho_L: float  # Liquid density (kg/m³)
    c_L: float  # Speed of sound in liquid (m/s)
    P_amb: float  # Ambient pressure (Pa)
    sigma_L: float  # Liquid surface tension (N/m)
    
    # Derived/Internal parameters
    R_break: float  # Break-up radius (m)
    sigma_break: float  # Surface tension at break-up (N/m)
    sigma_R0: float  # Surface tension at initial radius (N/m)
    vdw: float  # Van der Waals hard core radius (m)

    def __init__(
        self,
        R0,
        R_buckle,
        gamma,
        chi,
        mu_L,
        kappa_s,
        rho_L,
        c_L,
        P_amb,
        sigma_L,
    ):
        self.R0 = R0
        self.R_buckle = R_buckle
        self.gamma = gamma
        self.chi = chi
        self.mu_L = mu_L
        self.kappa_s = kappa_s
        self.rho_L = rho_L
        self.c_L = c_L
        self.P_amb = P_amb
        self.sigma_L = sigma_L

        self.R_break = 1.2 * self.R0
        self.sigma_break = ((self.R_break / self.R_buckle)**2 - 1) * self.chi
        self.sigma_R0 = self.chi * ((self.R0**2 / self.R_buckle**2) - 1)
        self.vdw = self.R0 / 5.61

    def surface_tension(self, R):
        """
        Marmottant model for surface tension.
        Smoothed to help solver convergence.
        """
        # if R<= self.buckle_radius: sigma = 0
        # elif R>= self.break_radius: sigma = self.sigma_liquid
        # else: sigma = self.comp_modulus*((R**2/self.buckle_radius**2) -1)

        # JAX implementation using jnp.where
        sigma_elastic = self.chi * ((R**2 / self.R_buckle**2) - 1)

        sigma = jnp.where(
            R <= self.R_buckle,
            0.0,
            jnp.where(
                R >= self.R_break,
                self.sigma_L, # Standard Marmottant: drops to liquid sigma after break
                sigma_elastic
            )
        )
        return sigma

    def get_scaled(self, units):
        return Bubble(
            R0=self.R0 / units.L_scale,
            R_buckle=self.R_buckle / units.L_scale,
            gamma=self.gamma,
            chi=self.chi / units.chi_scale,
            mu_L=self.mu_L / units.mu_scale,
            kappa_s=self.kappa_s / units.kappa_scale,
            rho_L=self.rho_L / units.rho_scale,
            c_L=self.c_L / units.vel_scale,
            P_amb=self.P_amb / units.P_scale,
            sigma_L=self.sigma_L / units.sigma_scale
        )

class Pulse(eqx.Module):
    """
    Parameters for the driving pulse.
    """
    freq: float
    pressure: float # Peak pressure (Pa)
    shape_func: Callable = eqx.field(static=True)
    n: int = 3
    phase: float = 0.0
    initial_time: float = 0.0
    cycle_num: float = 4.0
    
    def __call__(self, t):
        pulse_span = self.cycle_num / self.freq

        # Relative time from start of pulse
        tau = t - self.initial_time

        # Check if we are within the pulse duration
        in_pulse = (tau >= 0) & (tau <= pulse_span)

        # Hann window: 0.5 * (1 - cos(2*pi*tau/T))
        # This goes from 0 to 1 and back to 0 smoothly
        hann_window = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau / pulse_span))

        # Apply window (0 outside of pulse)
        # window = jnp.where(in_pulse, hann_window, 0.0)
        window = jnp.where(in_pulse, 1.0, 0.0)

        val = self.shape_func(t, self.freq, self.n, self.phase, self.initial_time)

        return val * self.pressure * window

    def get_scaled(self, units):
        return Pulse(
            freq=self.freq / units.freq_scale,
            pressure=self.pressure / units.P_scale,
            shape_func=self.shape_func,
            n=self.n,
            phase=self.phase,
            initial_time=self.initial_time / units.T_scale,
            cycle_num=self.cycle_num
        )

