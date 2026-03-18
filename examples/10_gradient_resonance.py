"""10 Gradient Descent to Resonance

Combining differentiability and batch sweeps.

This example shows how jbubble's differentiability can be used for
physics-based optimization. We:

1. Run a 2-D parameter sweep (frequency × bubble radius) to map out the
   expansion ratio landscape — the background heatmap.
2. Starting from an off-resonance point, use ``fit_parameters`` to jointly
   optimise driving frequency and bubble radius, converging to resonance.
3. Animate the optimization trajectory over the heatmap and save an MP4.

The key insight: because ``make_model`` receives the full params pytree,
frequency enters both the EoM (via R0 dynamics) and the pulse waveform from
a single optimizable parameter.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from jbubble import SaveSpec, fit_parameters, run_simulation
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import GompertzSurfaceTension, LipidShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.envelope import HannEnvelope
from jbubble.pulse.shapes import Sine
from jbubble.utils import GridSweep


# ============================================================================
# SHARED PHYSICS
# ============================================================================

PRESSURE = 10e3   # Pa — nonlinear regime gives a well-defined interior resonance peak
CYCLE_NUM = 5
T_SPAN = (0.0, 20e-6)  # s
SAVE_SPEC = SaveSpec(num_samples=200)

# Physical bounds for sigmoid-normalised optimisation
FREQ_MIN, FREQ_MAX = 0.1e6, 1.5e6   # Hz
R0_MIN, R0_MAX = 2.0e-6, 10.0e-6    # m


def _sigmoid_freq(x):
    return FREQ_MIN + (FREQ_MAX - FREQ_MIN) * jax.nn.sigmoid(x)


def _sigmoid_r0(x):
    return R0_MIN + (R0_MAX - R0_MIN) * jax.nn.sigmoid(x)


def _logit(frac):
    frac = jnp.clip(frac, 1e-6, 1 - 1e-6)
    return jnp.log(frac / (1 - frac))


def physical_to_params(freq, r0):
    return {
        "freq": _logit((freq - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)),
        "R0": _logit((r0 - R0_MIN) / (R0_MAX - R0_MIN)),
    }


def params_to_physical(params):
    return _sigmoid_freq(params["freq"]), _sigmoid_r0(params["R0"])


def soft_max(x, beta=1e6):
    """Smooth differentiable approximation of max via log-sum-exp.

    Avoids the argmax discontinuity of ``jnp.max``, which causes gradient
    spikes whenever the peak-radius timestep jumps between ODE output points.
    ``beta`` controls sharpness; at 1e6 / (typical radius ~µm) the error
    relative to the true max is negligible.
    """
    x_max = jnp.max(x)   # used only for numerical stability, not differentiated
    return x_max + jnp.log(jnp.mean(jnp.exp(beta * (x - x_max)))) / beta


def make_model(params):
    """Build (eom, pulse) from normalised params = [logit_freq, logit_r0]."""
    freq, r0 = params_to_physical(params)
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.07),
        shell=LipidShell(
            sigma=GompertzSurfaceTension(
                R_buckle_ratio=0.99,
                chi=0.38,
                sigma_rupture=72e-3,
            ),
            kappa_s=2.4e-9,
        ),
        medium=NewtonianMedium(mu=0.001),
        R0=r0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )
    pulse = ToneBurst(
        freq=freq,
        pressure=PRESSURE,
        shape=Sine(),
        cycle_num=CYCLE_NUM,
        envelope=HannEnvelope(),
    )
    return eom, pulse


# ============================================================================
# PARAMETER SWEEP (background heatmap)
# ============================================================================

def expansion_ratio_sweep(freq, R0):
    """Return max expansion ratio for (freq, R0).  vmappable."""
    eom, pulse = make_model(physical_to_params(freq, R0))
    result = run_simulation(eom, pulse, save_spec=SAVE_SPEC, t_span=T_SPAN)
    return soft_max(result.radius) / R0


# ============================================================================
# RESONANCE OPTIMISATION via fit_parameters
# ============================================================================

def find_resonance(init_freq, init_r0, n_steps, learning_rate):
    """Find the (freq, R0) pair that maximises expansion ratio.

    Uses ``fit_parameters`` with a maximisation loss and sigmoid-bounded
    params so the optimiser stays in the physically valid region.

    Returns
    -------
    freq_history, r0_history, expansion_history : jnp.ndarray
        Physical parameter and objective values at each step.
    fit_res : FitResult
    """
    freq_hist = []
    r0_hist = []
    expansion_hist = []

    # Add initial state before optimization
    eom, pulse = make_model(physical_to_params(init_freq, init_r0))
    init_result = run_simulation(eom, pulse, save_spec=SAVE_SPEC, t_span=T_SPAN)
    init_expansion = float(soft_max(init_result.radius) / init_result.state.R0[0])
    freq_hist.append(init_freq)
    r0_hist.append(init_r0)
    expansion_hist.append(init_expansion)

    def callback(_, params, loss_val):
        freq, r0 = params_to_physical(params)
        freq_hist.append(float(freq))
        r0_hist.append(float(r0))
        expansion_hist.append(-loss_val)  # loss is negated expansion ratio

    print("Starting resonance search from:")
    print(f"  freq = {init_freq/1e6:.3f} MHz,  R0 = {init_r0*1e6:.2f} µm")
    print(f"  lr = {learning_rate},  steps = {n_steps}\n")

    fit_res = fit_parameters(
        make_model=make_model,
        params0=physical_to_params(init_freq, init_r0),
        save_spec=SAVE_SPEC,
        t_span=T_SPAN,
        loss_fn=lambda result: -soft_max(result.radius) / result.state.R0[0],
        optimizer=optax.adam(learning_rate),
        n_steps=n_steps,
        step_callback=callback,
        log_every=1,
    )

    final_freq, final_r0 = params_to_physical(fit_res.params)
    print(f"\nConverged: freq = {float(final_freq)/1e6:.3f} MHz,  "
          f"R0 = {float(final_r0)*1e6:.2f} µm,  "
          f"expansion = {expansion_hist[-1]:.4f}")

    return (jnp.array(freq_hist), jnp.array(r0_hist),
            jnp.array(expansion_hist), fit_res)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_trajectory(freq_grid, r0_grid, expansion_grid,
                    freq_hist, r0_hist, init_freq, init_r0):
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.pcolormesh(freq_grid / 1e6, r0_grid * 1e6, expansion_grid,
                       shading="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="$R_\\mathrm{max}/R_0$")
    ax.plot(freq_hist / 1e6, r0_hist * 1e6, "w-", lw=2.5, alpha=0.9,
            label="Gradient descent")
    ax.plot(freq_hist / 1e6, r0_hist * 1e6, "wo", ms=5, alpha=0.7)
    ax.plot(init_freq / 1e6, init_r0 * 1e6, "ro", ms=14,
            markeredgecolor="w", markeredgewidth=2, label="Start", zorder=10)
    ax.plot(float(freq_hist[-1]) / 1e6, float(r0_hist[-1]) * 1e6, "g*", ms=20,
            markeredgecolor="w", markeredgewidth=2, label="Converged", zorder=10)
    ax.set_xlabel("Frequency (MHz)", fontsize=12)
    ax.set_ylabel("Initial Radius (µm)", fontsize=12)
    ax.set_title("Gradient Descent to Resonance", fontsize=14)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()




# ============================================================================
# MAIN
# ============================================================================

def main():
    # ── Step 1: parameter sweep ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Parameter sweep (background heatmap)")
    print("=" * 60)

    r0_values = jnp.linspace(2.0e-6, 10.0e-6, 100)
    freq_values = jnp.linspace(0.1e6, 1.5e6, 100)

    gs = GridSweep(
        fn=expansion_ratio_sweep,
        search_space={"R0": r0_values, "freq": freq_values},
        batch_size=256,
    )
    print(f"Running {gs.total_points} simulations …")
    t0 = time.time()
    flat = gs.collect()
    expansion_grid = gs.reshape(flat)  # shape (len(R0), len(freq))
    print(f"Sweep done in {time.time()-t0:.1f}s")

    # GridSweep sorts keys alphabetically: ["R0", "freq"]
    freq_grid, r0_grid = jnp.meshgrid(freq_values, r0_values)

    # ── Step 2: resonance search ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Gradient descent to resonance (via fit_parameters)")
    print("=" * 60)

    init_freq = 1.2e6
    init_r0 = 5.0e-6

    freq_hist, r0_hist, *_ = find_resonance(
        init_freq=init_freq,
        init_r0=init_r0,
        n_steps=50,
        learning_rate=0.3,
    )

    # ── Step 3: plot ─────────────────────────────────────────────────────────
    plot_trajectory(freq_grid, r0_grid, expansion_grid,
                    freq_hist, r0_hist, init_freq, init_r0)


if __name__ == "__main__":
    main()
