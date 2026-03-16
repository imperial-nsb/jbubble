"""Synthetic-data fitting — minimal jbubble example.

Demonstrates gradient-based parameter estimation:

  1. Simulate a "ground-truth" radius-time curve with known κ_s
  2. Start from an initial guess that is 10× too high
  3. Call fit_parameters — one line, no solver/adjoint boilerplate
  4. Compare the recovered κ_s to the true value

Run with:
    python examples/fit_simple.py
"""

from __future__ import annotations

import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jbubble import run_simulation
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import GompertzSurfaceTension, LipidShell
from jbubble.fitting import fit_parameters
from jbubble.metrics import normalised_mse_radius
from jbubble.pulse import HannEnvelope, ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.solver import SaveSpec

# ---------------------------------------------------------------------------
# Shared physical constants
# ---------------------------------------------------------------------------

R0 = 2.5e-6        # equilibrium radius         [m]
P_AMB = 101_325.0  # ambient pressure            [Pa]
RHO_L = 998.0      # liquid density              [kg/m³]
MU_L = 1e-3        # dynamic viscosity           [Pa·s]
C_L = 1481.0       # speed of sound              [m/s]

SAVE_SPEC = SaveSpec(num_samples=1024)
WINDOW_S = 20e-6

# ---------------------------------------------------------------------------
# Acoustic driving — 1 MHz, 5-cycle Hann-windowed burst at 150 kPa
# ---------------------------------------------------------------------------

pulse = ToneBurst(
    freq=1e6,
    pressure=150e3,
    shape=Sine(),
    cycle_num=5.0,
    envelope=HannEnvelope(),
)

# ---------------------------------------------------------------------------
# Model factory — parameterised by log(κ_s)
#
# Log-space keeps κ_s strictly positive and gives the optimiser a
# smoother landscape (κ_s spans several decades in practice).
# ---------------------------------------------------------------------------

KAPPA_S_TRUE = 5e-9   # ground-truth shell viscosity [kg/s]
KAPPA_S_INIT = 5e-8   # starting guess — 10× too high


def make_eom(log_kappa_s: jax.Array) -> KellerMiksis:
    shell = LipidShell(
        sigma=GompertzSurfaceTension(R_buckle_ratio=0.99, chi=0.5, sigma_break=72e-3),
        kappa_s=jnp.exp(log_kappa_s),
    )
    return KellerMiksis(
        gas=PolytropicGas(gamma=1.07),
        shell=shell,
        medium=NewtonianMedium(mu=MU_L),
        R0=R0,
        P_amb=P_AMB,
        rho_L=RHO_L,
        c_L=C_L,
    )


# ---------------------------------------------------------------------------
# 1. Ground-truth trajectory
# ---------------------------------------------------------------------------

target_result = jax.jit(run_simulation)(
    make_eom(jnp.log(jnp.array(KAPPA_S_TRUE))),
    pulse,
    save_spec=SAVE_SPEC,
    t_span=(0.0, WINDOW_S),
)
target = target_result.radius

print(f"Ground truth  κ_s = {KAPPA_S_TRUE:.1e} kg/s")
print(f"Initial guess κ_s = {KAPPA_S_INIT:.1e} kg/s  ({KAPPA_S_INIT / KAPPA_S_TRUE:.0f}× off)\n")

# ---------------------------------------------------------------------------
# 2. Fit — solver/adjoint choices handled automatically by fit_parameters
# ---------------------------------------------------------------------------

fit = fit_parameters(
    make_eom,
    pulse,
    params0=jnp.log(jnp.array(KAPPA_S_INIT)),
    save_spec=SAVE_SPEC,
    t_span=(0.0, WINDOW_S),
    loss_fn=lambda state: normalised_mse_radius(state.R, target, R0),
    optimizer=optax.adam(1e-2),
    n_steps=450,
)

kappa_s_fit = float(jnp.exp(fit.params))
print(f"\nFitted  κ_s = {kappa_s_fit:.3e} kg/s  (true = {KAPPA_S_TRUE:.1e})")

# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------

t_us = jnp.asarray(fit.result.ts) * 1e6

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("jbubble — gradient-based shell viscosity recovery", fontsize=11)

ax = axes[0]
ax.plot(t_us, target * 1e6, lw=2, label=f"Target  (κ_s = {KAPPA_S_TRUE:.0e} kg/s)")
ax.plot(
    t_us,
    fit.result.radius * 1e6,
    "--",
    lw=1.5,
    label=f"Fitted  (κ_s = {kappa_s_fit:.2e} kg/s)",
)
ax.set_xlabel("Time [µs]")
ax.set_ylabel("Radius [µm]")
ax.set_title("Radius-time curves")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.semilogy(fit.loss_history)
ax.set_xlabel("Adam step")
ax.set_ylabel("Normalised MSE")
ax.set_title("Optimisation convergence")
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("examples/fit_simple_result.png", dpi=150)
print("Figure saved → examples/fit_simple_result.png")
plt.show()
