"""Cattaneo & Supponen (2023) — gradient-based shell viscosity estimation.

Fits the shell dilatational viscosity κ_s for a single lipid-coated
microbubble by matching the peak normalised radial expansion predicted by
a Keller-Miksis / Marmottant model to a measured radius-time curve.

The optimisation is gradient-based: JAX differentiates through the entire
ODE integration (via diffrax's adjoint) to compute dL/d(log κ_s), and Adam
updates the parameter.

Reference:
    Cattaneo & Supponen, Soft Matter 2023, 19, 5925–5941.
    DOI: 10.1039/d3sm00871a

Usage:
    python examples/cattaneo/fit_shell_viscosity.py
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

import diffrax

from jbubble import (
    GompertzSurfaceTension,
    KellerMiksis,
    LipidShell,
    NewtonianMedium,
    PolytropicGas,
    Property,
    RectangularEnvelope,
    Sine,
    ToneBurst,
    run_simulation,
)
from jbubble.solver import SaveSpec, solve_eom

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Configuration — change these to switch between bubbles
# ---------------------------------------------------------------------------

CSV_NAME = "radius_curve_B.csv"
LABEL    = "Bubble B"

# ---------------------------------------------------------------------------
# Experimental constants — Cattaneo & Supponen (2023)
# ---------------------------------------------------------------------------

# Optics
PIXEL_SIZE_M = 160e-9   # 160 nm/px  (200× objective, HPV-X2 camera)
R_ERR_M      = 140e-9   # bright-field Fresnel bias to subtract [m]

# Liquid (deionised water at ~22 °C)
P_AMB   = 102.2e3   # ambient pressure        [Pa]
RHO_L   = 997.8     # density                 [kg/m³]
MU_L    = 9.54e-4   # dynamic viscosity       [Pa·s]
C_L     = 1481.0    # speed of sound          [m/s]
SIGMA_W = 72.8e-3   # water surface tension   [N/m]

# Gas core — air (replaces C4F10 within a few minutes, Kwan & Borden 2012)
GAMMA = 1.4

# Acoustic driving
F_DRIVE  = 1.5e6    # carrier frequency        [Hz]
P_DRIVE  = 40.0e3   # peak negative pressure   [Pa]
N_CYCLES = 20       # number of cycles

# Simulation
WINDOW_S = 25.6e-6  # matches HPV-X2 recording window [s]
N_SAVE   = 2048      # ODE output samples

# Shell modulus — fixed at a typical literature value.
# At resonance the peak expansion is insensitive to E_s (Cattaneo §4.2),
# so fitting κ_s alone is sufficient and better conditioned.
CHI_FIXED    = 0.6    # dilatational modulus E_s [N/m]
R_BUCKLE_RAT = 0.999  # R_buckle / R0 ≈ 1 → σ_0 ≈ 0 (tensionless at rest)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_radii(csv_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (time_s, radius_m, R0_m) from a radius-time CSV.

    Pixel radii are converted to metres, the 140 nm bright-field bias is
    subtracted, and R0 is estimated from the median of the first 10 frames
    (pre-pulse quiescent state).
    """
    df  = pd.read_csv(os.path.join(HERE, csv_name))
    t   = df["time_s"].to_numpy()
    r_m = df["radius_px"].to_numpy() * PIXEL_SIZE_M - R_ERR_M
    R0  = float(np.median(r_m[:10]))
    return t, r_m, R0


time_exp, radius_exp, R0 = load_radii(CSV_NAME)

# Interpolate the experimental radius onto the simulation time grid so the
# loss can compare the full waveform rather than just the peak.
ts_sim_grid    = np.linspace(0.0, WINDOW_S, N_SAVE)
radius_target  = jnp.array(np.interp(ts_sim_grid, time_exp, radius_exp))

print(f"{LABEL}:  R0 = {R0*1e6:.2f} µm   Rmax/R0−1 = {float(jnp.max(radius_target)/R0 - 1):.3f}")

# ---------------------------------------------------------------------------
# Acoustic pulse — hard-gated 20-cycle sinusoidal burst at 1.5 MHz / 40 kPa
# ---------------------------------------------------------------------------

def estimate_pulse_onset(t: np.ndarray, r: np.ndarray, n_baseline: int = 20) -> float:
    """Estimate pulse arrival time from the onset of bubble oscillation.

    Computes the noise floor from the first *n_baseline* frames (pre-pulse
    quiescent region) then finds the first frame whose deviation from R0
    exceeds 3 σ.  Returns that time minus half a carrier period so the
    onset sits just before the first measurable excursion.
    """
    r0        = np.median(r[:n_baseline])
    sigma     = np.std(r[:n_baseline])
    threshold = max(3.0 * sigma, PIXEL_SIZE_M)   # at least ±1 px
    for i in range(n_baseline, len(r)):
        if abs(r[i] - r0) > threshold:
            return max(0.0, float(t[i]) - 0.5 / F_DRIVE)
    return 0.0


def make_pulse(initial_time: float) -> ToneBurst:
    """Hard-gated 20-cycle sinusoidal burst starting at *initial_time*."""
    return ToneBurst(
        freq=F_DRIVE,
        pressure=P_DRIVE,
        shape=Sine(),
        cycle_num=float(N_CYCLES),
        envelope=RectangularEnvelope(),
        initial_time=initial_time,
    )


t_onset = estimate_pulse_onset(time_exp, radius_exp)
pulse   = make_pulse(t_onset)

print(f"{LABEL}:  pulse onset = {t_onset*1e6:.2f} µs")

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_eom(R0: float, log_kappa_s: jax.Array) -> KellerMiksis:
    """Assemble Keller-Miksis EOM with a Gompertz/Marmottant lipid shell.

    Parameters
    ----------
    R0 :
        Equilibrium bubble radius [m].
    log_kappa_s :
        Natural logarithm of the shell dilatational viscosity [log(kg/s)].
        Using log-space keeps the parameter well-scaled for the optimiser.
    """
    sigma = GompertzSurfaceTension(
        R_buckle_ratio=R_BUCKLE_RAT,
        chi=CHI_FIXED,
        sigma_break=SIGMA_W,
    )
    # Wrap in Property so as_property's float() converter is bypassed when
    # kappa_s is a traced JAX array (e.g. during jax.grad).
    shell  = LipidShell(sigma=sigma, kappa_s=Property(val=jnp.exp(log_kappa_s)))
    gas    = PolytropicGas(gamma=GAMMA)
    medium = NewtonianMedium(mu=MU_L)
    return KellerMiksis(
        gas=gas, shell=shell, medium=medium,
        R0=R0, P_amb=P_AMB, rho_L=RHO_L, c_L=C_L,
    )

# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------

# Tsit5 (explicit RK) + BacksolveAdjoint: explicit solvers avoid the
# ill-conditioned linear-system adjoint that Kvaerno5 produces, and
# BacksolveAdjoint reintegrates backward rather than differentiating through
# all stored steps.
SOLVER        = diffrax.Tsit5()
ADJOINT       = diffrax.BacksolveAdjoint()
PID           = diffrax.PIDController(rtol=1e-4, atol=1e-8)
FIT_SAVE_SPEC = SaveSpec(num_samples=N_SAVE)

# ---------------------------------------------------------------------------
# Loss function — squared error in peak normalised expansion
# ---------------------------------------------------------------------------


def loss_fn(log_kappa_s: jax.Array, R0: float, radius_target: jax.Array,
            pulse: ToneBurst) -> jax.Array:
    """Mean squared waveform error in normalised radius: mean((R_sim − R_exp)²/R0²)."""
    eom = make_eom(R0, log_kappa_s)
    sol = solve_eom(
        eom, pulse,
        t_span=(0.0, WINDOW_S),
        dt0=1e-9,
        save_spec=FIT_SAVE_SPEC,
        solver=SOLVER,
        stepsize_controller=PID,
        adjoint=ADJOINT,
        max_steps=200_000,
    )
    return jnp.mean(((sol.ys.R - radius_target) / R0) ** 2)


# JIT-compile the primal + gradient together for efficiency.
loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=0))

# ---------------------------------------------------------------------------
# Optimisation loop
# ---------------------------------------------------------------------------

def fit(
    R0:            float,
    radius_target: jax.Array,
    pulse:         ToneBurst,
    label:         str,
    log_ks0: float = float(np.log(5e-9)),   # initial guess ≈ 5×10⁻⁹ kg/s
    n_steps: int   = 400,
    lr:      float = 0.01,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (κ_s [kg/s], loss_history, kappa_s_history) from Adam optimisation."""
    param = jnp.array(log_ks0)
    opt   = optax.adam(lr)
    state = opt.init(param)

    loss_hist = np.empty(n_steps)
    ks_hist   = np.empty(n_steps)

    print(f"\n[{label}]  R0 = {R0*1e6:.2f} µm")
    print(f"  {'step':>4}  {'loss':>12}  {'κ_s [kg/s]':>14}")

    for step in range(n_steps):
        val, grad  = loss_and_grad(param, R0, radius_target, pulse)
        updates, state = opt.update(grad, state)
        param = optax.apply_updates(param, updates)

        loss_hist[step] = float(val)
        ks_hist[step]   = float(jnp.exp(param))

        if step % 25 == 0 or step == n_steps - 1:
            print(f"  {step:>4}  {float(val):>12.4e}  {ks_hist[step]:>14.3e}")

    return float(jnp.exp(param)), loss_hist, ks_hist


kappa_s, loss_hist, ks_hist = fit(R0, radius_target, pulse, LABEL)

print(f"\n{'─'*45}")
print(f"{LABEL}:  κ_s = {kappa_s:.3e} kg/s  (E_s = {CHI_FIXED} N/m fixed)")
print(f"{'─'*45}")
print("Expected range from paper: 1×10⁻⁹ – 1×10⁻⁸ kg/s")

# ---------------------------------------------------------------------------
# Final simulation with fitted parameter
# ---------------------------------------------------------------------------

eom_fit = make_eom(R0, float(np.log(kappa_s)))
result  = run_simulation(
    eom_fit, pulse,
    save_spec=FIT_SAVE_SPEC,
    window_s=WINDOW_S,
    dt0=1e-9,
    solver=SOLVER,
    max_steps=200_000,
)
radius_sim = result.radius
ts_sim     = result.ts

# ---------------------------------------------------------------------------
# Plot — paper-style: R/R0 curve + driving pulse + convergence
# ---------------------------------------------------------------------------

# Simulation and experiment share the same time axis [0, WINDOW_S]:
# initial_time on the pulse ensures the bubble sits at equilibrium before
# the pulse arrives, matching the quiescent pre-pulse region in the data.
t_plot  = np.linspace(0.0, WINDOW_S, 2000)
p_pulse = np.array([float(pulse(jnp.array(t))) for t in t_plot])

PIXEL_ERR = PIXEL_SIZE_M  # ±1 pixel positional uncertainty

fig = plt.figure(figsize=(10, 8))
fig.suptitle(f"Cattaneo & Supponen (2023) — {LABEL} shell viscosity fit",
             fontsize=11, y=0.98)

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[3, 2],
    hspace=0.45, wspace=0.35,
    left=0.09, right=0.97, top=0.93, bottom=0.07,
)

# ---- R/R0 − 1 vs time (top, full width) -----------------------------------
ax_r = fig.add_subplot(gs[0, :])

norm_exp    = radius_exp / R0 - 1.0
norm_target = np.asarray(radius_target) / R0 - 1.0
norm_sim    = np.asarray(radius_sim) / R0 - 1.0
err_band    = PIXEL_ERR / R0

ax_r.fill_between(time_exp * 1e6, norm_exp - err_band, norm_exp + err_band,
                  color="C0", alpha=0.15, linewidth=0)
ax_r.plot(time_exp * 1e6, norm_exp, "o", ms=2.5, alpha=0.4,
          color="C0", label="Experiment (raw)")
ax_r.plot(ts_sim_grid * 1e6, norm_target, "-", lw=1.0, alpha=0.7,
          color="C0", label="Experiment (interp.)")
ax_r.plot(np.asarray(ts_sim) * 1e6, norm_sim, "-", lw=1.8, color="C1",
          label=(rf"Fit: $\kappa_s={kappa_s:.2e}$ kg/s, "
                 rf"$E_s={CHI_FIXED}$ N/m (fixed)"))
ax_r.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
ax_r.set_ylabel(r"$(R/R_0) - 1$", fontsize=9)
ax_r.set_xlabel("Time [µs]", fontsize=9)
rmax_target = float(jnp.max(radius_target) / R0 - 1.0)
ax_r.set_title(
    f"{LABEL}   ($R_0 = {R0*1e6:.2f}$ µm,  "
    rf"$R_\mathrm{{max}}/R_0 - 1 = {rmax_target:.3f}$)",
    fontsize=9,
)
ax_r.legend(fontsize=8)
ax_r.grid(True, alpha=0.25)

# ---- Driving pulse waveform (bottom-left) ---------------------------------
ax_p = fig.add_subplot(gs[1, 0])

ax_p.plot(t_plot * 1e6, p_pulse * 1e-3, color="C2", lw=1.2)
ax_p.fill_between(t_plot * 1e6, p_pulse * 1e-3, alpha=0.15, color="C2")
ax_p.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
ax_p.set_xlabel("Time [µs]", fontsize=9)
ax_p.set_ylabel("Pressure [kPa]", fontsize=9)
ax_p.set_title(f"Driving pulse  ({F_DRIVE/1e6:.1f} MHz, {P_DRIVE/1e3:.0f} kPa, "
               f"{N_CYCLES} cycles)", fontsize=9)
ax_p.grid(True, alpha=0.25)

# ---- Convergence (bottom-right) -------------------------------------------
ax_c = fig.add_subplot(gs[1, 1])

steps = np.arange(len(loss_hist))
ax_c.semilogy(steps, loss_hist, color="C1", lw=1.2)
ax_c.set_xlabel("Adam step", fontsize=9)
ax_c.set_ylabel(r"Loss  $\langle(R_\mathrm{sim} - R_\mathrm{exp})^2\rangle / R_0^2$",
                fontsize=8)
ax_c.set_title("Optimisation convergence", fontsize=9)
ax_c.grid(True, alpha=0.25, which="both")

out_path = os.path.join(HERE, "fit_result.png")
fig.savefig(out_path, dpi=150)
print(f"\nFigure saved → {out_path}")
plt.show()
