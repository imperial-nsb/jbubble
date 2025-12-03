import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from physics import Bubble, Pulse
from solver import solve_bubble
import shapes

def run_simulation(R0, pressure):
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
        freq=200e3,
        pressure=pressure,
        cycle_num=3,
        initial_time=0,
        n=3,
        shape_func=shapes.pulse_sine # pulse1 is sine
    )

    # Visualize the pulse
    t_end = pulse.cycle_num / pulse.freq
    t_vis = jnp.linspace(0, t_end, 1000)
    p_vis = jax.vmap(pulse)(t_vis)

    plt.figure(figsize=(10, 4))
    plt.plot(np.array(t_vis) * 1e6, np.array(p_vis) / 1e3)
    plt.xlabel('Time (μs)')
    plt.ylabel('Pressure (kPa)')
    plt.title('Acoustic Pulse')
    plt.grid(True)
    plt.savefig('pulse_visualization.png')
    plt.show()
    print("Pulse visualization saved to pulse_visualization.png")

    sol = solve_bubble(bubble, pulse)

    # Visualize the solution
    ts = sol.ts
    Rs = sol.ys[:, 0]
    
    plt.figure(figsize=(10, 4))
    plt.plot(np.array(ts) * 1e6, np.array(Rs) * 1e6)
    plt.xlabel('Time (μs)')
    plt.ylabel('Radius (μm)')
    plt.title('Bubble Radius vs Time')
    plt.grid(True)
    plt.savefig('bubble_radius.png')
    plt.show()
    print("Bubble radius visualization saved to bubble_radius.png")

    # Extract metrics
    Rs = sol.ys[:, 0]
    # Filter out NaNs if any (though solver should handle it)
    Rs = jnp.nan_to_num(Rs, nan=R0)

    max_R = jnp.max(Rs)
    min_R = jnp.min(Rs)
    
    max_ratio = max_R / R0
    min_ratio = R0 / min_R
    max_min_ratio = max_R / min_R
    
    return max_R, max_ratio, min_R, min_ratio, max_min_ratio

def main():
    print("Running bubble simulations with JAX/Diffrax...")

    # # Grid
    # radii = jnp.linspace(0.5e-6, 5e-6, 2)
    # pressures = jnp.linspace(50e3, 200e3, 3)
    
    # # Create meshgrid for vmap
    # # We want to iterate over all combinations
    # R_grid, P_grid = jnp.meshgrid(radii, pressures)
    
    # # Flatten for vmap
    # R_flat = R_grid.ravel()
    # P_flat = P_grid.ravel()
    
    # # JIT compile the vmapped function
    # # Note: diffrax.diffeqsolve is JIT-compatible
    # vmapped_run = eqx.filter_jit(jax.vmap(run_simulation))
    
    # print(f"Simulating {len(R_flat)} combinations...")
    # results = vmapped_run(R_flat, P_flat)

    # Single run
    R0 = 5e-6
    pressure = 200e3
    results = run_simulation(R0, pressure)

    max_R, max_ratio, min_R, min_ratio, max_min_ratio = results

    # # Reshape results
    # shape = R_grid.shape
    # max_ratio_grid = max_ratio.reshape(shape)
    
    # # Plotting (using matplotlib, so convert to numpy)
    # max_ratio_np = np.array(max_ratio_grid)
    # radii_np = np.array(radii)
    # pressures_np = np.array(pressures)
    
    # plt.figure(figsize=(10, 8))
    # plt.imshow(
    #     max_ratio_np, 
    #     extent=[radii_np.min()*1e6, radii_np.max()*1e6, pressures_np.min()/1000, pressures_np.max()/1000],
    #     origin='lower', # Note: exact.py used 'upper' but had pressures reversed in extent? 
    #     # exact.py: extent=[min, max, max, min], origin='upper'. 
    #     # This means top row is max pressure.
    #     # My meshgrid: P_grid rows correspond to P values. 
    #     # If I use origin='lower', bottom row is min pressure.
    #     cmap='inferno',
    #     aspect='auto'
    # )
    # plt.colorbar(label='Maximum Radius / Initial Radius')
    # plt.xlabel('Initial Radius (μm)')
    # plt.ylabel('Pressure (kPa)')
    # plt.title('Expansion Ratio (JAX/Diffrax)')
    # plt.savefig('heatmap_jax.png')
    # print("Heatmap saved to heatmap_jax.png")

if __name__ == "__main__":
    main()
