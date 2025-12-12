
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

class ParametricFourierShape(shapes.PulseShape):
    coeffs: jax.Array 

    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __call__(self, t, freq, phase, initial_time):
        t = t - initial_time
        m = jnp.arange(1, self.coeffs.shape[0] + 1)
        
        def term(i, c):
             return c * jnp.sin(2.0 * jnp.pi * i * freq * t - phase)

        val = jnp.sum(jax.vmap(term)(m, self.coeffs), axis=0)
        norm = jnp.sum(jnp.abs(self.coeffs)) + 1e-6
        return val / norm

def run_optimization():
    print("Starting Co-Design Selective Triggering Optimization...")
    print("optimizing: [Pulse A (Shape+Freq)] + [Pulse B (Shape+Freq)] + [Bubble A Params] + [Bubble B Params]")
    
    units = Units()
    save_spec = SaveSpec(num_samples=200) 
    
    # --- Genome Definition ---
    # Total dims: 26
    # 0-9:   Pulse A Shape (10)
    # 10:    Pulse A Freq (1)
    # 11:    Bubble A Radius (1)
    # 12:    Bubble A Chi (1)
    # 13-22: Pulse B Shape (10)
    # 23:    Pulse B Freq (1)
    # 24:    Bubble B Radius (1)
    # 25:    Bubble B Chi (1)
    
    NUM_SHAPE = 10
    GENOME_SIZE = 2 * (NUM_SHAPE + 1 + 2) # 26
    
    # Parameter Bounds / Scaling
    MIN_FREQ = 0.2e6
    MAX_FREQ = 0.8e6
    
    MIN_RAD = 1.5e-6
    MAX_RAD = 7.0e-6 # More typical contrast agent range
    
    MIN_CHI = 0.2  # ~0.0 is unrealistic (zero tension). 0.1 is soft lipid.
    MAX_CHI = 0.8  # 1.0 is quite stiff. Default is ~0.38.

    pressure_fixed = 300e3

    @jax.jit
    def decode_genome(genome):
        # Helper to map [0,1] or standard normal to physical ranges
        # We assume genome is somewhat centered around 0 with std 1 (CMA-ES), 
        # but we can use sigmoid to map to bounds.
        
        def map_val(val, min_v, max_v):
            return min_v + (max_v - min_v) * jax.nn.sigmoid(val)
        
        # Pulse A
        shape_a_coeffs = genome[0:NUM_SHAPE]
        freq_a = map_val(genome[NUM_SHAPE], MIN_FREQ, MAX_FREQ)
        r0_a = map_val(genome[NUM_SHAPE+1], MIN_RAD, MAX_RAD)
        chi_a = map_val(genome[NUM_SHAPE+2], MIN_CHI, MAX_CHI)
        
        # Pulse B
        offset = NUM_SHAPE + 3
        shape_b_coeffs = genome[offset:offset+NUM_SHAPE]
        freq_b = map_val(genome[offset+NUM_SHAPE], MIN_FREQ, MAX_FREQ)
        r0_b = map_val(genome[offset+NUM_SHAPE+1], MIN_RAD, MAX_RAD)
        chi_b = map_val(genome[offset+NUM_SHAPE+2], MIN_CHI, MAX_CHI)
        
        return (shape_a_coeffs, freq_a, r0_a, chi_a), (shape_b_coeffs, freq_b, r0_b, chi_b)

    @jax.jit
    def fitness_fn(genome):
        (s_a, f_a, r_a, c_a), (s_b, f_b, r_b, c_b) = decode_genome(genome)
        
        # Construct objects
        shape_obj_a = ParametricFourierShape(s_a)
        shape_obj_b = ParametricFourierShape(s_b)
        
        pulse_a = Pulse(freq=f_a, pressure=200e3, shape=shape_obj_a, cycle_num=5, initial_time=1e-6)
        pulse_b = Pulse(freq=f_b, pressure=200e3, shape=shape_obj_b, cycle_num=5, initial_time=1e-6)
        
        bubble_a = Bubble(R0=r_a, chi=c_a)
        bubble_b = Bubble(R0=r_b, chi=c_b)
        
        # Simulations (Window adapts to lower freq)
        min_freq = jnp.minimum(f_a, f_b)
        window = 10.0 / min_freq # Ensure enough time
        
        # Cross-Talk Matrix:
        # P_A -> B_A (Target: High)
        # P_A -> B_B (Target: Low)
        # P_B -> B_A (Target: Low)
        # P_B -> B_B (Target: High)
        
        # 1. Pulse A on both
        res_aa = run_simulation(bubble_a, pulse_a, units=units, save_spec=save_spec, window_s=window)
        res_ab = run_simulation(bubble_b, pulse_a, units=units, save_spec=save_spec, window_s=window)
        
        # 2. Pulse B on both
        res_ba = run_simulation(bubble_a, pulse_b, units=units, save_spec=save_spec, window_s=window)
        res_bb = run_simulation(bubble_b, pulse_b, units=units, save_spec=save_spec, window_s=window)
        
        max_aa = jnp.max(res_aa.radius) / r_a
        max_ab = jnp.max(res_ab.radius) / r_b
        max_ba = jnp.max(res_ba.radius) / r_a
        max_bb = jnp.max(res_bb.radius) / r_b
        
        # Objectives
        # We want diagonals > 2.0, off-diagonals < 1.5
        
        # Score A: Pulse A good?
        score_a_target = jnp.maximum(0.0, 2.5 - max_aa) # Penalty if < 2.5
        score_a_avoid  = jnp.maximum(0.0, max_ab - 1.2) # Penalty if > 1.2
        
        # Score B: Pulse B good?
        score_b_target = jnp.maximum(0.0, 2.5 - max_bb) # Penalty if < 2.5
        score_b_avoid  = jnp.maximum(0.0, max_ba - 1.2) # Penalty if > 1.2
        
        # Constraint: Distinct bubbles?
        # Penalty if radii are too close
        radius_diff = jnp.abs(r_a - r_b)
        score_distinct = jnp.maximum(0.0, 1.0e-6 - radius_diff) * 1000 # Large penalty if diff < 1um
        
        return score_a_target + score_a_avoid + score_b_target + score_b_avoid + score_distinct, (max_aa, max_ab, max_ba, max_bb)

    # --- Optimization ---
    rng = jax.random.PRNGKey(42)
    solution_template = jnp.zeros(GENOME_SIZE)
    strategy = CMA_ES(population_size=64, solution=solution_template)
    es_params = strategy.default_params
    
    state = strategy.init(rng, solution_template, es_params)
    
    num_gens = 100
    print(f"Optimizing for {num_gens} generations...")
    
    best_fitness_history = []
    
    for gen in range(num_gens):
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_gen, state, es_params)
        
        loss_vals, metrics = jax.vmap(fitness_fn)(x)
        # Metrics is a tuple of arrays, we only need losses for update
        
        state, _ = strategy.tell(rng_eval, x, loss_vals, state, es_params)
        
        best_l = state.best_fitness
        best_fitness_history.append(best_l)
        
        if gen % 10 == 0:
            print(f"Gen {gen}: Best Loss = {best_l:.4f}")

    print(f"Final Loss: {state.best_fitness:.4f}")
    best_genome = state.best_solution
    
    # --- Analysis ---
    (s_a, f_a, r_a, c_a), (s_b, f_b, r_b, c_b) = decode_genome(best_genome)
    print("\n--- Optimized System ---")
    print("Bubble A:")
    print(f"  Radius: {r_a*1e6:.2f} um")
    print(f"  Chi:    {c_a:.2f}")
    print("Bubble B:")
    print(f"  Radius: {r_b*1e6:.2f} um")
    print(f"  Chi:    {c_b:.2f}")
    print("Pulse A:")
    print(f"  Freq:   {f_a/1e6:.2f} MHz")
    print("Pulse B:")
    print(f"  Freq:   {f_b/1e6:.2f} MHz")
    
    # Re-run best to plotting
    # We need to construct result objects.
    # We'll just perform the 4 sims again.
    
    shape_obj_a = ParametricFourierShape(s_a)
    shape_obj_b = ParametricFourierShape(s_b)
    
    pulse_a = Pulse(freq=f_a, pressure=pressure_fixed, shape=shape_obj_a, cycle_num=5, initial_time=1e-6)
    pulse_b = Pulse(freq=f_b, pressure=pressure_fixed, shape=shape_obj_b, cycle_num=5, initial_time=1e-6)
    bubble_a = Bubble(R0=r_a, chi=c_a)
    bubble_b = Bubble(R0=r_b, chi=c_b)
    
    min_freq = min(f_a, f_b)
    window = 10.0 / min_freq
    save_spec_plot = SaveSpec(num_samples=1000)
    
    # Run 4 combinations
    res_aa = run_simulation(bubble_a, pulse_a, units=units, save_spec=save_spec_plot, window_s=window)
    res_ab = run_simulation(bubble_b, pulse_a, units=units, save_spec=save_spec_plot, window_s=window)
    res_ba = run_simulation(bubble_a, pulse_b, units=units, save_spec=save_spec_plot, window_s=window)
    res_bb = run_simulation(bubble_b, pulse_b, units=units, save_spec=save_spec_plot, window_s=window)
    
    max_aa = jnp.max(res_aa.radius) / r_a
    max_ab = jnp.max(res_ab.radius) / r_b
    max_ba = jnp.max(res_ba.radius) / r_a
    max_bb = jnp.max(res_bb.radius) / r_b
    
    print("\n--- Cross-Talk Matrix (R_max/R_0) ---")
    print(f"P_A -> B_A: {max_aa:.2f} (Target > 2.5)")
    print(f"P_A -> B_B: {max_ab:.2f} (Avoid  < 1.2)")
    print(f"P_B -> B_A: {max_ba:.2f} (Avoid  < 1.2)")
    print(f"P_B -> B_B: {max_bb:.2f} (Target > 2.5)")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row 1: Pulse A
    # Left: Pulse Shape
    # Right: Responses
    ax = axes[0, 0]
    ts_a = res_aa.ts
    p_vals_a = pulse_a(ts_a)
    ax.plot(ts_a*1e6, p_vals_a/1e3, 'k')
    ax.set_title(f"Pulse A (Freq={f_a/1e6:.2f} MHz)")
    ax.set_ylabel("Pressure (kPa)")
    
    ax = axes[0, 1]
    ax.plot(ts_a*1e6, res_aa.radius / r_a, 'b', label=f"Bubble A ({r_a*1e6:.1f}um)")
    ax.plot(ts_a*1e6, res_ab.radius / r_b, 'r--', label=f"Bubble B ({r_b*1e6:.1f}um)")
    ax.axhline(2.0, color='gray', linestyle=':')
    ax.set_title("Response to Pulse A")
    ax.set_ylabel("R/R0")
    ax.legend()

    # Row 2: Pulse B
    ax = axes[1, 0]
    ts_b = res_bb.ts
    p_vals_b = pulse_b(ts_b)
    ax.plot(ts_b*1e6, p_vals_b/1e3, 'k')
    ax.set_title(f"Pulse B (Freq={f_b/1e6:.2f} MHz)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_xlabel("Time (us)")
    
    ax = axes[1, 1]
    ax.plot(ts_b*1e6, res_ba.radius / r_a, 'b--', label=f"Bubble A")
    ax.plot(ts_b*1e6, res_bb.radius / r_b, 'r', label=f"Bubble B")
    ax.axhline(2.0, color='gray', linestyle=':')
    ax.set_title("Response to Pulse B")
    ax.set_ylabel("R/R0")
    ax.set_xlabel("Time (us)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("co_design_results.png")
    print("\nSaved results to co_design_results.png")
    plt.show()

if __name__ == "__main__":
    run_optimization()
