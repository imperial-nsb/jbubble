from typing import Tuple
import jax.numpy as jnp
import diffrax
from diffrax import TextProgressMeter
from physics import Bubble, Pulse

def bubble_equation(t, state, args: Tuple[Bubble, Pulse]):
    """
    The ODE function for the bubble dynamics.
    state: [R, R_dot]
    args: (Bubble, Pulse)
    """
    R, R_dot = state
    bubble, pulse = args

    # Unpack bubble parameters
    R0 = bubble.R0
    vdw = bubble.vdw
    gamma = bubble.gamma
    c_L = bubble.c_L
    P_amb = bubble.P_amb
    rho_L = bubble.rho_L
    mu_L = bubble.mu_L
    kappa_s = bubble.kappa_s
    sigma_R0 = bubble.sigma_R0

    # Surface tension at current radius
    sigma = bubble.surface_tension(R)

    # Driving pressure
    P_drive = pulse(t)

    # Terms
    # Gas pressure (Van der Waals)
    P_gas0 = P_amb + 2 * sigma_R0 / R0  # Ambient + initial surface tension term
    P_gas = P_gas0 * ((R0**3 - vdw**3) / (R**3 - vdw**3)) ** gamma

    # Viscous and Surface Tension terms
    P_surf = 2 * sigma / R
    P_visc = 4 * mu_L * R_dot / R
    P_surf_visc = 4 * kappa_s * R_dot / (R**2)

    # Damping term in gas pressure (radiation damping approximation?)
    # (1 - 3 * gamma * R_dot * R**3 / (c_L * (R**3 - vdw**3)))
    damping_term = 1 - (3 * gamma * (R**3) * R_dot) / (c_L * (R**3 - vdw**3))

    forces = (P_gas * damping_term) - P_surf - P_visc - P_surf_visc - P_drive - P_amb

    R_ddot = (forces / rho_L - 1.5 * R_dot**2) / R

    return jnp.stack([R_dot, R_ddot])

def solve_bubble(bubble: Bubble, pulse: Pulse, t_span=None, dt0=1e-9):
    """
    Solves the bubble dynamics ODE.
    """
    if t_span is None:
        # Default to pulse duration + some extra
        t_span = (
            0.0,
            pulse.initial_time + 0.4 * (pulse.cycle_num / pulse.freq),
        )

    term = diffrax.ODETerm(bubble_equation)
    # Use a stiff solver (Kvaerno5) as bubble dynamics can be stiff near collapse
    solver = diffrax.Kvaerno5()
    # solver = diffrax.Euler()

    t0, t1 = t_span
    y0 = jnp.array([bubble.R0, 0.0])

    # Save at high resolution for plotting
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 10000))

    # PID controller for step size
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=(bubble, pulse),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=300_000, # Increased max steps even further
        progress_meter=TextProgressMeter(),
        throw=True,
    )

    return sol
