"""05 Parameter Fitting

Differentiable Microbubble Dynamics.

One of the unique features of jbubble is its differentiability.
You can compute the gradient of the final simulation result
with respect to any physical parameter (radii, shell viscosity,
elasticity, etc.) and use it to fit models to experimental data.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
from jbubble import run_simulation, fit_parameters, SaveSpec
from jbubble.bubble.eom import EquationOfMotion, ModifiedRayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import (
    GompertzSurfaceTension,
    LipidShell,
)
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.simulation import SimulationResult

R0 = 2e-6  # [m]


# 1. Define the forward model factory
# This builds an EquationOfMotion from the parameter we want to fit (chi).
def make_eom(params: dict) -> EquationOfMotion:
    # Marmottant shell elasticity (chi) is what we will estimate.
    sigma = GompertzSurfaceTension(
        R_buckle_ratio=0.98,
        chi=params["chi"],
        sigma_rupture=0.072,
    )
    return ModifiedRayleighPlesset(
        gas=PolytropicGas(gamma=1.07),
        shell=LipidShell(sigma=sigma, kappa_s=2.5e-9),
        medium=NewtonianMedium(mu=0.001),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


# Common settings
pulse = ToneBurst(
    freq=1e6,
    pressure=50e3,
    shape=Sine(),
    cycle_num=5,
)
save_spec = SaveSpec(num_samples=256)

# 2. Generate "Experimental Data" (Ground truth)
# We pretend chi=0.5 is the real physical value we don't know.
true_chi = 0.5
ground_truth_res = run_simulation(
    make_eom({"chi": true_chi}), pulse, save_spec=save_spec
)
# Add Gaussian noise to simulate realistic measurement uncertainty.
key = jr.PRNGKey(42)
radius_span = ground_truth_res.radius.max() - ground_truth_res.radius.min()
noise_std = 0.05 * radius_span  # 5% of radius span
noise = jr.normal(key, shape=ground_truth_res.radius.shape) * noise_std
target_radius = ground_truth_res.radius + noise


# 3. Define the Loss Function
# The solver will pass a SimulationResult to this function.
def loss_fn(result: SimulationResult) -> jax.Array:
    # Mean Squared Error between simulated radius and ground truth.
    # Normalizing by the equilibrium radius improves optimization stability.
    return jnp.mean((result.state.R - target_radius) ** 2) / R0


# 4. Run Gradient-Based Optimization
print("Starting gradient-based fitting of shell elasticity (chi)...")
print(f"Ground truth chi = {true_chi:.4f}")

# Initial guess chi = 0.1
initial_guess = {"chi": jnp.array(0.1)}

fit_res = fit_parameters(
    make_model=lambda p: (make_eom(p), pulse),
    params0=initial_guess,
    save_spec=save_spec,
    loss_fn=loss_fn,
    optimizer=optax.adam(learning_rate=0.03),
    n_steps=100,
)

fitted_chi = float(fit_res.params["chi"])
print(f"Final fitted chi = {fitted_chi:.4f}")

# 5. Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot loss history
ax1.plot(fit_res.loss_history, color="blue", lw=2)
ax1.set_yscale("log")
ax1.set_xlabel("Optimization Step")
ax1.set_ylabel("MSE Loss")
ax1.set_title("Optimization Convergence")
ax1.grid(True, which="both", alpha=0.3)

# Plot radius curves
ax2.plot(
    ground_truth_res.ts * 1e6, target_radius * 1e6, "k.", label="Target Data", alpha=0.4
)
ax2.plot(
    fit_res.result.ts * 1e6,
    fit_res.result.radius * 1e6,
    "r-",
    label="Fitted Model",
    lw=2,
)
ax2.set_xlabel("Time (µs)")
ax2.set_ylabel("Radius (µm)")
ax2.set_title(f"Target vs Fitted Curve (chi = {fitted_chi:.3f})")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
