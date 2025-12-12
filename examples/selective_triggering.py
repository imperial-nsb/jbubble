
import jax
import jax.numpy as jnp
from jax import config
import matplotlib.pyplot as plt
import evosax
from evosax.algorithms import CMA_ES

from jbubble import (
    Units,
    SaveSpec,
    Bubble,
    Pulse,
    run_simulation,
    shapes,
)

config.update("jax_enable_x64", True)

# Define a custom PulseShape that uses Fourier coefficients
class ParametricFourierShape(shapes.PulseShape):
    coeffs: jax.Array  # The parameters we will optimize (num_terms,)

    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __call__(self, t, freq, phase, initial_time):
        t = t - initial_time
        m = jnp.arange(1, self.coeffs.shape[0] + 1)
        
        # Simple sine series sum: sum(coeff[i] * sin(2*pi*i*f*t))
        # This allows constructing arbitrary periodic waveforms
        def term(i, c):
             return c * jnp.sin(2.0 * jnp.pi * i * freq * t - phase)

        # Vectorize over terms and sum
        val = jnp.sum(jax.vmap(term)(m, self.coeffs), axis=0)
        
        # Normalize by sum of absolute coefficients to keep amplitude roughly bounded
        norm = jnp.sum(jnp.abs(self.coeffs)) + 1e-6
        return val / norm


def run_optimization():
    print("Starting selective bubble triggering optimization...")
    
    units = Units()
    # We only need the max radius, so we don't need to save all time steps
    # But currently run_simulation returns everything. We can use a minimal SaveSpec.
    save_spec = SaveSpec(num_samples=100) # Keep it small for speed? Or does it matter for JIT?
    
    # --- Problem Setup ---
    # Bubble A: R0 = 2.0 um
    # Bubble B: R0 = 4.0 um
    bubble_a = Bubble(R0=2.0e-6)
    bubble_b = Bubble(R0=4.0e-6)
    
    # Pulse constraints
    base_freq = 1.0e6
    pressure = 150e3 # 150 kPa
    num_coeffs = 10
    
    # Define the fitness function for Evosax
    # We optimize 'coeffs'
    
    @jax.jit
    def fitness_fn(coeffs):
        # Construct the shape
        shape = ParametricFourierShape(coeffs)
        
        pulse = Pulse(
            freq=base_freq,
            pressure=pressure,
            shape=shape,
            cycle_num=5,
            initial_time=1e-6
        )
        
        # Run simulations
        # We need a small window. 5 cycles at 1MHz = 5us. 
        # Add buffer. 10us total window.
        res_a = run_simulation(bubble_a, pulse, units=units, save_spec=save_spec, window_s=10e-6)
        res_b = run_simulation(bubble_b, pulse, units=units, save_spec=save_spec, window_s=10e-6)
        
        # Metrics: Max expansion ratio
        max_r_a = jnp.max(res_a.radius) / bubble_a.R0
        max_r_b = jnp.max(res_b.radius) / bubble_b.R0
        
        return max_r_a, max_r_b

    # We want to perform TWO optimizations:
    # 1. Trigger A (MaxR > 2.0), Ignore B (MaxR < 2.0)
    # 2. Trigger B (MaxR > 2.0), Ignore A (MaxR < 2.0)
    
    # --- Strategy ---
    # We use CMA-ES from evosax
    rng = jax.random.PRNGKey(42)
    # New API: passes solution template instead of num_dims?
    # Based on inspect output: (self, population_size: int, solution: Any, ...)
    solution_template = jnp.zeros(num_coeffs)
    strategy = CMA_ES(population_size=32, solution=solution_template)
    es_params = strategy.default_params
    
    num_generations = 50
    
    # --- Optimization 1: Target A, Avoid B ---
    print("\n--- Optimizing: Target A, Avoid B ---")
    
    @jax.jit
    def loss_target_a(coeffs):
        r_a, r_b = fitness_fn(coeffs)
        # We want r_a > 2.0, r_b < 2.0
        # Loss = ReLU(2.0 - r_a) + ReLU(r_b - 1.5) 
        # (Using 1.5 as a safer "avoid" threshold to handle oscillation, or just use 1.9?)
        # Let's use soft penalties.
        
        loss_a = jnp.maximum(0.0, 2.2 - r_a) # Penalty if A < 2.2
        loss_b = jnp.maximum(0.0, r_b - 1.8) # Penalty if B > 1.8
        
        return loss_a + loss_b

    state_a = strategy.init(rng, jnp.zeros(num_coeffs), es_params)
    
    for gen in range(num_generations):
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        x, state_a = strategy.ask(rng_gen, state_a, es_params)
        
        # Evaluate population
        loss_vals = jax.vmap(loss_target_a)(x)
        
        # Update strategy
        state_a, _ = strategy.tell(rng_eval, x, loss_vals, state_a, es_params)
        
        if gen % 10 == 0:
            best_l = state_a.best_fitness
            print(f"Gen {gen}: Best Loss = {best_l:.4f}")

    best_coeffs_a = state_a.best_solution
    print(f"Final Loss (Target A): {state_a.best_fitness:.4f}")
    
    
    # --- Optimization 2: Target B, Avoid A ---
    print("\n--- Optimizing: Target B, Avoid A ---")
    
    @jax.jit
    def loss_target_b(coeffs):
        r_a, r_b = fitness_fn(coeffs)
        # We want r_b > 2.2, r_a < 1.8
        
        loss_b = jnp.maximum(0.0, 2.2 - r_b) # Penalty if B < 2.2
        loss_a = jnp.maximum(0.0, r_a - 1.8) # Penalty if A > 1.8
        
        return loss_a + loss_b

    state_b = strategy.init(rng, jnp.zeros(num_coeffs), es_params)
    
    for gen in range(num_generations):
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        x, state_b = strategy.ask(rng_gen, state_b, es_params)
        
        loss_vals = jax.vmap(loss_target_b)(x)
        state_b, _ = strategy.tell(rng_eval, x, loss_vals, state_b, es_params)
        
        if gen % 10 == 0:
            best_l = state_b.best_fitness
            print(f"Gen {gen}: Best Loss = {best_l:.4f}")

    best_coeffs_b = state_b.best_solution
    print(f"Final Loss (Target B): {state_b.best_fitness:.4f}")

    # --- Analysis & Plotting ---
    print("\nSimulating best solutions...")
    res_a_target_a, res_b_target_a = check_solution(best_coeffs_a, bubble_a, bubble_b, units, base_freq, pressure)
    res_a_target_b, res_b_target_b = check_solution(best_coeffs_b, bubble_a, bubble_b, units, base_freq, pressure)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row 1: Target A (Avoid B)
    t_axis = res_a_target_a.ts * 1e6 # microseconds
    
    # Plot Pulse
    ax = axes[0, 0]
    pulse_obj_a = Pulse(freq=base_freq, pressure=pressure, shape=ParametricFourierShape(best_coeffs_a), cycle_num=5, initial_time=1e-6)
    # We can just use the pressure from the result or recompute pulse
    # But result doesn't store full pulse array unless we did save_spec?
    # Actually run_simulation uses the pulse object.
    # Let's reconstruct the pulse array for plotting
    p_values = pulse_obj_a(res_a_target_a.ts)
    ax.plot(t_axis, p_values / 1e3, color='black')
    ax.set_title("Optimized Pulse (Target A)")
    ax.set_ylabel("Pressure (kPa)")
    ax.grid(True, alpha=0.3)
    
    # Plot Response
    ax = axes[0, 1]
    ax.plot(t_axis, res_a_target_a.radius / bubble_a.R0, label="Bubble A (Target)", color='blue')
    ax.plot(t_axis, res_b_target_a.radius / bubble_b.R0, label="Bubble B (Avoid)", color='red', linestyle='--')
    ax.axhline(2.0, color='gray', linestyle=':', label='Threshold (2.0)')
    ax.set_title("Bubble Response")
    ax.set_ylabel("expansion ratio (R/R0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 2: Target B (Avoid A)
    # Plot Pulse
    ax = axes[1, 0]
    pulse_obj_b = Pulse(freq=base_freq, pressure=pressure, shape=ParametricFourierShape(best_coeffs_b), cycle_num=5, initial_time=1e-6)
    p_values = pulse_obj_b(res_a_target_b.ts)
    ax.plot(t_axis, p_values / 1e3, color='black')
    ax.set_title("Optimized Pulse (Target B)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_xlabel("Time (µs)")
    ax.grid(True, alpha=0.3)
    
    # Plot Response
    ax = axes[1, 1]
    ax.plot(t_axis, res_a_target_b.radius / bubble_a.R0, label="Bubble A (Avoid)", color='blue', linestyle='--')
    ax.plot(t_axis, res_b_target_b.radius / bubble_b.R0, label="Bubble B (Target)", color='red')
    ax.axhline(2.0, color='gray', linestyle=':', label='Threshold (2.0)')
    ax.set_title("Bubble Response")
    ax.set_ylabel("expansion ratio (R/R0)")
    ax.set_xlabel("Time (µs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("selective_triggering_results.png")
    print("\nResults saved to selective_triggering_results.png")
    plt.show()

def check_solution(coeffs, bubble_a, bubble_b, units, freq, pressure):
    shape = ParametricFourierShape(coeffs)
    pulse = Pulse(
        freq=freq,
        pressure=pressure,
        shape=shape,
        cycle_num=5,
        initial_time=1e-6
    )
    # Use higher resolution for final check/plot
    save_spec = SaveSpec(num_samples=1000)
    res_a = run_simulation(bubble_a, pulse, units=units, save_spec=save_spec, window_s=10e-6)
    res_b = run_simulation(bubble_b, pulse, units=units, save_spec=save_spec, window_s=10e-6)
    
    print(f"  Bubble A Max R/R0: {jnp.max(res_a.radius)/bubble_a.R0:.2f}")
    print(f"  Bubble B Max R/R0: {jnp.max(res_b.radius)/bubble_b.R0:.2f}")
    return res_a, res_b


if __name__ == "__main__":
    run_optimization()
