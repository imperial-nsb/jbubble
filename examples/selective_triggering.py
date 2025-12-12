
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


import jax
import jax.numpy as jnp
from jax import config
import matplotlib.pyplot as plt
import evosax
from evosax.algorithms import CMA_ES
import equinox as eqx
import diffrax

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

# --- Genome Definition with Equinox ---
class Parameter(eqx.Module):
    min_val: float
    max_val: float

class SystemGenome(eqx.Module):
    # System Constraints
    pulse_shape_n: int = eqx.field(static=True)

    freq_bounds: Parameter
    cycle_bounds: Parameter
    radius_bounds: Parameter
    chi_bounds: Parameter
    kappa_bounds: Parameter
    gamma_bounds: Parameter

    def __init__(self):
        self.pulse_shape_n = 2

        # Define bounds here
        self.freq_bounds   = Parameter(0.75e6, 0.75e6)
        self.cycle_bounds  = Parameter(10.0, 13.0) # 1 to 10 cycles
        self.radius_bounds = Parameter(1.5e-6, 4.0e-6)
        self.chi_bounds    = Parameter(0.38, 0.38)
        self.kappa_bounds  = Parameter(2.4e-9, 2.4e-9)
        self.gamma_bounds  = Parameter(1.07, 1.07)

    @property
    def genome_size(self):
        # Per Pulse-Bubble Pair:
        # Shape(N) + Freq(1) + Cycles(1) + Radius(1) + Chi(1) + Kappa(1) + Gamma(1)
        # = N + 6
        # Total = 2 * (N + 6)
        return 2 * (self.pulse_shape_n + 6)

    def decode(self, genome):
        # Map genome [-inf, inf] (CMA-ES) to [0, 1] then to bounds
        # We assume input is ~standard normal, we use sigmoid to bound it.
        
        def map_p(val, param):
            return param.min_val + (param.max_val - param.min_val) * jax.nn.sigmoid(val)
        
        N = self.pulse_shape_n
        
        # --- Channel A ---
        # Indices:
        # 0..N-1: Shape
        # N: Freq
        # N+1: Cycles
        # N+2: Radius
        # N+3: Chi
        # N+4: Kappa
        # N+5: Gamma
        cutoff_a = N + 6
        
        g_a = genome[0:cutoff_a]
        shape_a = g_a[0:N] # Coefficients are unbounded
        freq_a  = map_p(g_a[N],   self.freq_bounds)
        cyc_a   = map_p(g_a[N+1], self.cycle_bounds)
        r_a     = map_p(g_a[N+2], self.radius_bounds)
        c_a     = map_p(g_a[N+3], self.chi_bounds)
        k_a     = map_p(g_a[N+4], self.kappa_bounds)
        gam_a   = map_p(g_a[N+5], self.gamma_bounds)
        
        # --- Channel B ---
        g_b = genome[cutoff_a:]
        shape_b = g_b[0:N]
        freq_b  = map_p(g_b[N],   self.freq_bounds)
        cyc_b   = map_p(g_b[N+1], self.cycle_bounds)
        r_b     = map_p(g_b[N+2], self.radius_bounds)
        c_b     = map_p(g_b[N+3], self.chi_bounds)
        k_b     = map_p(g_b[N+4], self.kappa_bounds)
        gam_b   = map_p(g_b[N+5], self.gamma_bounds)
        
        return (shape_a, freq_a, cyc_a, r_a, c_a, k_a, gam_a), \
               (shape_b, freq_b, cyc_b, r_b, c_b, k_b, gam_b)

def run_optimization():
    print("Starting Robust Co-Design Selective Triggering Optimization...")
    
    # 1. Initialize System Spec
    sys_spec = SystemGenome()
    print(f"Genome Size: {sys_spec.genome_size}")
    
    units = Units()
    save_spec = SaveSpec(num_samples=1000) 
    pressure_fixed = 400e3

    @jax.jit
    def fitness_fn(genome):
        (s_a, f_a, cyc_a, r_a, c_a, k_a, g_a), (s_b, f_b, cyc_b, r_b, c_b, k_b, g_b) = sys_spec.decode(genome)
        
        # Construct objects
        shape_obj_a = ParametricFourierShape(s_a)
        shape_obj_b = ParametricFourierShape(s_b)
        
        pulse_a = Pulse(freq=f_a, pressure=pressure_fixed, shape=shape_obj_a, cycle_num=cyc_a, initial_time=1e-6, apply_hann=True)
        pulse_b = Pulse(freq=f_b, pressure=pressure_fixed, shape=shape_obj_b, cycle_num=cyc_b, initial_time=1e-6, apply_hann=True)

        bubble_a = Bubble(R0=r_a, chi=c_a, kappa_s=k_a, gamma=g_a)
        bubble_b = Bubble(R0=r_b, chi=c_b, kappa_s=k_b, gamma=g_b)

        # Adaptive Window
        # Window needs to be long enough for the longest pulse + ringdown
        dur_a = cyc_a / f_a
        dur_b = cyc_b / f_b
        window = jnp.maximum(dur_a, dur_b) * 2.0 

        # --- Simulations with Convergence Check ---
        def run_checked(bub, pul):
            res = run_simulation(bub, pul, units=units, save_spec=save_spec, window_s=window)
            # Penalize if solver failed
            valid = res.converged
            # Safe radius extraction (if NaN, results in penalty)
            # We add a huge penalty term if not valid.
            penalty = jnp.where(valid, 0.0, 1e6) 
            return res.radius, penalty

        rad_aa, pen_aa = run_checked(bubble_a, pulse_a)
        rad_ab, pen_ab = run_checked(bubble_b, pulse_a)
        rad_ba, pen_ba = run_checked(bubble_a, pulse_b)
        rad_bb, pen_bb = run_checked(bubble_b, pulse_b)
        
        # Max Expansion Ratios
        max_aa = jnp.nanmax(rad_aa) / r_a
        max_ab = jnp.nanmax(rad_ab) / r_b
        max_ba = jnp.nanmax(rad_ba) / r_a
        max_bb = jnp.nanmax(rad_bb) / r_b
        
        # Objectives
        # We want diagonals > 2.2, off-diagonals < 1.2
        
        # Score A: Pulse A good?
        score_a_target = jnp.maximum(0.0, 3.0 - max_aa)
        score_a_avoid  = jnp.maximum(0.0, max_ab - 1.2)
        
        # Score B: Pulse B good?
        score_b_target = jnp.maximum(0.0, 3.0 - max_bb)
        score_b_avoid  = jnp.maximum(0.0, max_ba - 1.2)
        
        total_penalty = pen_aa + pen_ab + pen_ba + pen_bb
        
        # No distinct radii penalty!
        
        return score_a_target + score_a_avoid + score_b_target + score_b_avoid + total_penalty, (max_aa, max_ab, max_ba, max_bb)

    # --- Optimization ---
    rng = jax.random.PRNGKey(42)
    solution_template = jnp.zeros(sys_spec.genome_size)
    strategy = CMA_ES(population_size=64, solution=solution_template)
    es_params = strategy.default_params

    state = strategy.init(rng, solution_template, es_params)
    
    num_gens = 100
    print(f"Optimizing for {num_gens} generations...")
    
    best_fitness_history = []
    best_genome_history = []
    
    for gen in range(num_gens):
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_gen, state, es_params)
        
        loss_vals, metrics = jax.vmap(fitness_fn)(x)
        state, _ = strategy.tell(rng_eval, x, loss_vals, state, es_params)
        
        best_l = state.best_fitness
        best_fitness_history.append(best_l)
        best_genome_history.append(state.best_solution)
        
        if gen % 10 == 0:
            print(f"Gen {gen}: Best Loss = {best_l:.4f}")

    print(f"Final Loss: {state.best_fitness:.4f}")
    best_genome = state.best_solution
    
    # --- Analysis ---
    (s_a, f_a, cyc_a, r_a, c_a, k_a, g_a), (s_b, f_b, cyc_b, r_b, c_b, k_b, g_b) = sys_spec.decode(best_genome)
    
    print("\n--- Optimized System ---")
    print("Bubble A:")
    print(f"  Radius: {r_a*1e6:.2f} um")
    print(f"  Chi:    {c_a:.2f}")
    print(f"  Kappa:  {k_a:.2e}")
    print(f"  Gamma:  {g_a:.2f}")
    print("Bubble B:")
    print(f"  Radius: {r_b*1e6:.2f} um")
    print(f"  Chi:    {c_b:.2f}")
    print(f"  Kappa:  {k_b:.2e}")
    print(f"  Gamma:  {g_b:.2f}")
    print("Pulse A:")
    print(f"  Freq:   {f_a/1e6:.2f} MHz")
    print(f"  Cycles: {cyc_a:.1f}")
    print("Pulse B:")
    print(f"  Freq:   {f_b/1e6:.2f} MHz")
    print(f"  Cycles: {cyc_b:.1f}")
    
    # ----------------------------
    # New: Parameter Evolution Viz
    # ----------------------------
    # We extract the scalar parameters from the history and normalized them to [0,1]
    # (i.e. just sigmoid(genome_val))
    
    import numpy as np
    history_stack = np.array(best_genome_history) # [Gens, GenomeSize]
    
    # Identify indices for scalar params
    # A: N..N+5
    # B: (N+6)+N .. (N+6)+N+5 = 2N+6 .. 2N+11
    # Let's map them explicitly
    
    N = sys_spec.pulse_shape_n
    
    # Labels and Indices
    param_map = [
        ("Freq A",  N),
        ("Cyc A",   N+1),
        ("Rad A",   N+2),
        ("Chi A",   N+3),
        ("Kap A",   N+4),
        ("Gam A",   N+5),
        ("Freq B",  (N+6) + N),
        ("Cyc B",   (N+6) + N+1),
        ("Rad B",   (N+6) + N+2),
        ("Chi B",   (N+6) + N+3),
        ("Kap B",   (N+6) + N+4),
        ("Gam B",   (N+6) + N+5),
    ]
    
    evol_data = [] # shape [NumParams, Gens]
    labels = []
    
    for lbl, idx in param_map:
        # Get trace for this parameter
        raw_trace = history_stack[:, idx]
        # Normalize: sigmoid
        norm_trace = 1.0 / (1.0 + np.exp(-raw_trace)) 
        evol_data.append(norm_trace)
        labels.append(lbl)
        
    evol_data = np.array(evol_data) # [Params, Gens]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(evol_data, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(label='Normalized Value (0=Min, 1=Max)')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Generation')
    plt.title('Parameter Evolution')
    plt.tight_layout()
    plt.savefig('parameter_evolution.png')
    print("Saved evolution plot to parameter_evolution.png")
    
    # ----------------------------
    
    # Plotting
    shape_obj_a = ParametricFourierShape(s_a)
    shape_obj_b = ParametricFourierShape(s_b)
    
    pulse_a = Pulse(freq=f_a, pressure=pressure_fixed, shape=shape_obj_a, cycle_num=cyc_a, initial_time=1e-6, apply_hann=True)
    pulse_b = Pulse(freq=f_b, pressure=pressure_fixed, shape=shape_obj_b, cycle_num=cyc_b, initial_time=1e-6, apply_hann=True)
    bubble_a = Bubble(R0=r_a, chi=c_a, kappa_s=k_a, gamma=g_a)
    bubble_b = Bubble(R0=r_b, chi=c_b, kappa_s=k_b, gamma=g_b)
    
    dur_a = cyc_a / f_a
    dur_b = cyc_b / f_b
    window = jnp.maximum(dur_a, dur_b) * 3.0
    save_spec_plot = SaveSpec(num_samples=1000)
    
    res_aa = run_simulation(bubble_a, pulse_a, units=units, save_spec=save_spec_plot, window_s=window)
    res_ab = run_simulation(bubble_b, pulse_a, units=units, save_spec=save_spec_plot, window_s=window)
    res_ba = run_simulation(bubble_a, pulse_b, units=units, save_spec=save_spec_plot, window_s=window)
    res_bb = run_simulation(bubble_b, pulse_b, units=units, save_spec=save_spec_plot, window_s=window)
    
    max_aa = jnp.max(res_aa.radius) / r_a
    max_ab = jnp.max(res_ab.radius) / r_b
    max_ba = jnp.max(res_ba.radius) / r_a
    max_bb = jnp.max(res_bb.radius) / r_b
    
    print("\n--- Cross-Talk Matrix (R_max/R_0) ---")
    print(f"P_A -> B_A: {max_aa:.2f} (Target > 2.2)")
    print(f"P_A -> B_B: {max_ab:.2f} (Avoid  < 1.2)")
    print(f"P_B -> B_A: {max_ba:.2f} (Avoid  < 1.2)")
    print(f"P_B -> B_B: {max_bb:.2f} (Target > 2.2)")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row 1: Pulse A
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
    ax.legend(loc='upper right')

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
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("co_design_results.png")
    print("\nSaved results to co_design_results.png")
    # plt.show() # Commented out for headless execution

if __name__ == "__main__":
    run_optimization()
