import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jbubble import (
    Units,
    SaveSpec,
    Bubble,
    Pulse,
    run_simulation,
)
import jbubble.shapes as shapes


def run_2d_sweep(x_values, y_values, kernel_func, x_name="X", y_name="Y"):
    """
    Generic function to run a 2D parameter sweep.

    Args:
        x_values: Array of values for the first parameter (x-axis).
        y_values: Array of values for the second parameter (y-axis).
        kernel_func: A function that takes (x, y) and returns the simulation result.
        x_name: Name of the x parameter for logging.
        y_name: Name of the y parameter for logging.

    Returns:
        results_grid: The simulation results reshaped into a 2D grid (y, x).
        x_grid: Meshgrid for x values.
        y_grid: Meshgrid for y values.
    """
    print(f"Preparing sweep: {y_name} vs {x_name}...")

    # Create meshgrid
    # Note: meshgrid with 'xy' indexing (default) returns:
    # x_grid[i, j] depends on j (x-axis)
    # y_grid[i, j] depends on i (y-axis)
    x_grid, y_grid = jnp.meshgrid(x_values, y_values)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()

    # Vectorize the kernel function
    # We assume kernel_func takes (x, y) as arguments
    vectorized_run = jax.vmap(kernel_func)

    # JIT compile the vectorized function for performance
    jit_vectorized_run = jax.jit(vectorized_run)

    print(f"Running sweep with {x_flat.size} simulations...")
    start_time = time.time()

    # Run the simulations
    results_flat = jit_vectorized_run(x_flat, y_flat)

    # Block until ready to measure actual execution time
    _ = results_flat.radius.block_until_ready()

    end_time = time.time()
    duration = end_time - start_time
    print(f"Sweep completed in {duration:.2f} seconds ({duration/x_flat.size:.2e} s/sim)")
    
    # Reshape results back to 2D grid
    # We use jax.tree.map to handle the PyTree structure of the result object
    results_grid = jax.tree.map(
        lambda x: x.reshape(*x_grid.shape, *x.shape[1:]), 
        results_flat
    )

    return results_grid, x_grid, y_grid

def plot_heatmap(x_grid, y_grid, data, x_label, y_label, title, cbar_label):
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_grid, y_grid, data, shading='auto', cmap='viridis')
    plt.colorbar(label=cbar_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.gca().invert_yaxis() # Optional, depends on preference
    plt.show()

def main():
    units = Units()
    save_spec = SaveSpec(num_samples=1000)

    # --- Sweep 1: Frequency vs Initial Radius ---
    print("\n--- Sweep 1: Frequency vs Initial Radius ---")

    def freq_r0_kernel(freq, r0):
        bubble = Bubble(R0=r0)
        pulse = Pulse(
            freq=freq,
            pressure=100e3,
            shape=shapes.Asymmetrical(),
            cycle_num=5,
            initial_time=1e-6
        )
        return run_simulation(bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6)

    r0_values = jnp.linspace(1.0e-6, 10.0e-6, 50)
    freq_values = jnp.linspace(0.1e6, 1.5e6, 50)

    # Note: run_2d_sweep takes (x_values, y_values). 
    # We want Freq on X and R0 on Y.
    res_1, freq_grid_1, r0_grid_1 = run_2d_sweep(
        freq_values, r0_values, freq_r0_kernel, x_name="Frequency", y_name="R0"
    )

    expansion_ratio_1 = res_1.radius.max(axis=-1) / res_1.bubble.R0
    plot_heatmap(
        freq_grid_1 / 1e6, 
        r0_grid_1 * 1e6, 
        expansion_ratio_1,
        "Frequency (MHz)", 
        "Initial Radius (µm)", 
        "Max Expansion Ratio: Freq vs R0",
        "$R_{max}/R_0$"
    )

    # --- Sweep 2: Initial Radius vs Shell Elasticity (Chi) ---
    print("\n--- Sweep 2: Initial Radius vs Shell Elasticity (Chi) ---")

    # Fixed frequency for this sweep
    fixed_freq = 800e3

    def r0_chi_kernel(r0, chi):
        bubble = Bubble(R0=r0, chi=chi)
        pulse = Pulse(
            freq=fixed_freq,
            pressure=100e3,
            shape=shapes.Asymmetrical(),
            cycle_num=5,
            initial_time=1e-6
        )
        return run_simulation(bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6)

    r0_values_2 = jnp.linspace(1.0e-6, 10.0e-6, 50)
    chi_values = jnp.linspace(0.0, 1.0, 50)

    # We want R0 on X and Chi on Y
    res_2, r0_grid_2, chi_grid_2 = run_2d_sweep(
        r0_values_2, chi_values, r0_chi_kernel, x_name="R0", y_name="Chi"
    )

    expansion_ratio_2 = res_2.radius.max(axis=-1) / res_2.bubble.R0
    plot_heatmap(
        r0_grid_2 * 1e6, 
        chi_grid_2, 
        expansion_ratio_2,
        "Initial Radius (µm)", 
        r"Shell Elasticity $\chi$ (N/m)", 
        f"Max Expansion Ratio: R0 vs Chi (Freq={fixed_freq/1e6} MHz)",
        "$R_{max}/R_0$"
    )

    # --- Sweep 3: Driving Pressure vs Initial Radius ---
    print("\n--- Sweep 3: Driving Pressure vs Initial Radius ---")

    def pressure_r0_kernel(pressure, r0):
        bubble = Bubble(R0=r0)
        pulse = Pulse(
            freq=fixed_freq,
            pressure=pressure,
            shape=shapes.Asymmetrical(),
            cycle_num=5,
            initial_time=1e-6
        )
        return run_simulation(bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6)

    pressure_values = jnp.linspace(10e3, 500e3, 50) # 10 kPa to 500 kPa
    r0_values_3 = jnp.linspace(1.0e-6, 10.0e-6, 50)

    # We want Pressure on X and R0 on Y
    res_3, p_grid_3, r0_grid_3 = run_2d_sweep(
        pressure_values, r0_values_3, pressure_r0_kernel, x_name="Pressure", y_name="R0"
    )

    expansion_ratio_3 = res_3.radius.max(axis=-1) / res_3.bubble.R0
    plot_heatmap(
        p_grid_3 / 1e3, 
        r0_grid_3 * 1e6, 
        expansion_ratio_3,
        "Driving Pressure (kPa)", 
        "Initial Radius (µm)", 
        f"Max Expansion Ratio: Pressure vs R0 (Freq={fixed_freq/1e6} MHz)",
        "$R_{max}/R_0$"
    )

if __name__ == "__main__":
    main()
