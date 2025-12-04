import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
import diffrax
from physics import Bubble, Pulse, Units
from solver import solve_bubble
import shapes
import time

def run_simulation(R0, pressure, f0):
    units = Units()

    bubble = Bubble(
        R0=R0,
        R_buckle=0.99 * R0,
        gamma=1.07,
        chi=0.38,
        mu_L=0.00089,
        kappa_s=2.4e-9,
        rho_L=1000.0,
        c_L=1498.0,
        P_amb=101.3e3,
        sigma_L=72e-3
    )
    
    # Setup Pulse
    
    pulse = Pulse(
        freq=f0,
        pressure=pressure,
        cycle_num=10,
        initial_time=1e-6,
        n=3,
        shape_func=shapes.pulse_sine # pulse1 is sine
    )

    bubble = bubble.get_scaled(units)
    pulse = pulse.get_scaled(units)

    # Time the solver - first call includes JIT compilation
    
    num_runs = 5
    times = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        sol = solve_bubble(bubble, pulse, dt0=1e-3)
        sol.ys.block_until_ready()  # Ensure JAX computation completes
        end = time.perf_counter()
        times.append(end - start)
        print(f"Run {i+1}: {times[-1]*1000:.2f} ms")
    
    print(f"\nFirst run (with JIT): {times[0]*1000:.2f} ms")
    print(f"Subsequent runs (post-JIT) avg: {np.mean(times[1:])*1000:.2f} ms")
    ts = sol.ts
    Rs = sol.ys[:, 0]

    p_vis = jax.vmap(pulse)(ts)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.set_ylabel('Driving Pressure (kPa)', color='tab:blue')
    ax1.plot(np.array(ts), np.array(p_vis), label="Driving Pressure", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Radius (μm)', color='tab:orange')
    ax2.plot(np.array(ts), np.array(Rs), label="Radius / Time", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


    # Extract metrics
    Rs = sol.ys[:, 0]
    # Filter out NaNs if any (though solver should handle it)
    # Rs = jnp.nan_to_num(Rs, nan=R0)

    max_R = jnp.max(Rs)
    min_R = jnp.min(Rs)

    max_ratio = max_R / bubble.R0
    min_ratio = bubble.R0 / min_R
    max_min_ratio = max_R / min_R

    return max_R, max_ratio, min_R, min_ratio, max_min_ratio

def main():
    # print("Running bubble simulations with JAX/Diffrax...")

    # Single run
    R0 = 4e-6
    pressure = 1000e3
    f0 = 800e3
    results = run_simulation(R0, pressure, f0)

if __name__ == "__main__":
    main()
