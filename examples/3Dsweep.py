#!/usr/bin/env python3
"""
Created on Sun Jan 25 01:00:29 2026

@author: ssm321
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from jbubble.bubble import MarmottantGompertz
from jbubble.units import Units
from jbubble.pulse import Pulse
from jbubble.solver import SaveSpec
from jbubble.simulation import run_simulation
import jbubble.shapes as shapes

# Enable 64-bit precision for better stability in physical simulations
from jax import config

config.update("jax_enable_x64", True)

units = Units()
save_spec = SaveSpec(num_samples=1000)
run_simulation_jit = jax.jit(run_simulation)

Bubble = MarmottantGompertz


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
    x_grid, y_grid = jnp.meshgrid(x_values, y_values)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()

    # Vectorize and JIT (your pattern)
    vectorized_run = jax.vmap(kernel_func)
    jit_vectorized_run = jax.jit(vectorized_run)

    print(f"Running sweep with {x_flat.size} simulations...")
    start_time = time.time()

    results_flat = jit_vectorized_run(x_flat, y_flat)

    # Block until ready to measure actual execution time
    _ = results_flat.radius.block_until_ready()

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"Sweep completed in {duration:.2f} seconds ({duration / x_flat.size:.2e} s/sim)"
    )

    # Reshape PyTree results into 2D grid
    results_grid = jax.tree.map(
        lambda x: x.reshape(*x_grid.shape, *x.shape[1:]), results_flat
    )
    return results_grid, x_grid, y_grid


def edges_from_centers(centers):
    centers = jnp.asarray(centers)
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    left_edge = centers[0] - (midpoints[0] - centers[0])
    right_edge = centers[-1] + (centers[-1] - midpoints[-1])
    return jnp.concatenate((left_edge[None], midpoints, right_edge[None]), axis=0)


def run_3d_sweep(
    x_values, y_values, z_values, kernel3d, x_name="X", y_name="Y", z_name="Z"
):
    """
    Run a 3D sweep over (x, y, z). For each z0 in z_values, run a 2D sweep
    using your run_2d_sweep, then compute the scalar heat R_max/R0.

    Returns:
        heats: (nz, ny, nx) array of scalars (e.g., max expansion ratio).
        x_grid, y_grid: grids for plotting.
    """
    heats_list = []
    cavit_list = []
    x_grid_ref = None
    y_grid_ref = None

    for i, z0 in enumerate(z_values):
        label = z0.__name__ if callable(z0) else f"{z_name}={float(z0):.4g}"

        if callable(z0):
            print(f"\n--- 2D slice at {z_name} " + label + "---")

        else:
            print(f"\n--- 2D slice at {z_name}={float(z0):.4g} ---")

        # Build a 2D kernel that closes over z0
        def kernel2d(x, y):
            return kernel3d(x, y, z0)

        # Run 2D sweep at this z
        res, x_grid, y_grid = run_2d_sweep(
            x_values, y_values, kernel2d, x_name=x_name, y_name=y_name
        )

        # Scalar metric: max expansion ratio per cell (ny, nx)
        heat2d = res.radius.max(axis=-1) / res.bubble.R0
        cav2d = (heat2d > 2).astype(int)

        heats_list.append(heat2d)
        cavit_list.append(cav2d)
        if x_grid_ref is None:
            x_grid_ref, y_grid_ref = x_grid, y_grid

    heats = jnp.stack(heats_list, axis=0)  # (nz, ny, nx)
    cavit_heats = jnp.stack(cavit_list, axis=0)
    return heats, x_grid_ref, y_grid_ref, cavit_heats


def plot_stacked_planes(
    x_centers,
    y_centers,
    z_centers,
    heats,
    cavit_heats,
    title="Stacked planes of $R_{max}/R_0$",
    x_label="X",
    y_label="Y",
    z_label="Z",
):
    """
    x_centers: (nx,)
    y_centers: (ny,)
    z_centers: (nz,)
    heats: (nz, ny, nx)
    """
    # Global color normalization
    vmin = float(jnp.nanmin(heats))
    vmax = float(jnp.nanmax(heats))
    cmap = plt.get_cmap("turbo", 256)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Face grids (ny+1, nx+1)
    x_edges = edges_from_centers(x_centers)
    y_edges = edges_from_centers(y_centers)
    Xe, Ye = jnp.meshgrid(x_edges, y_edges, indexing="xy")

    # Convert to NumPy for Matplotlib
    Xe = np.asarray(Xe)
    Ye = np.asarray(Ye)
    heats_np = np.asarray(heats)
    z_centers_np = np.asarray(z_centers)

    # Create figure/axis
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    z_labels = []
    z_discrete_ticks = np.arange(0, len(z_centers), dtype=float)

    # Plot each z plane
    for k, z0 in enumerate(z_centers_np):
        facecolors = cmap(norm(heats_np[k]))  # (ny, nx, 4)
        Ze = np.full_like(Xe, float(k), dtype=float)

        ax.plot_surface(
            Xe,
            Ye,
            Ze,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            edgecolor="none",
            antialiased=False,
            shade=False,
            alpha=0.8,
        )

        if callable(z0):
            label = z0.__name__
            z_labels.append(label)

    # Aesthetics (cubic aspect, ticks at centers)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xticks(np.asarray(x_centers))
    ax.set_yticks(np.asarray(y_centers))
    ax.set_zticks(np.linspace(1, len(z_centers), len(z_centers)))

    ax.set_xticks(np.linspace(x_centers.min(), x_centers.max(), 5))
    ax.set_yticks(np.linspace(y_centers.min(), y_centers.max(), 5))

    if z_labels:
        ax.set_zticks(z_discrete_ticks)
        ax.set_zticklabels(z_labels)
    else:
        ax.set_zticks(np.linspace(1, len(z_centers), len(z_centers)))

    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (1, 0, 0, 1)  # red
        axis._axinfo["grid"]["linewidth"] = 0.1
        axis._axinfo["grid"]["linestyle"] = "-"

    # Shared colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(heats_np)
    fig.colorbar(mappable, ax=ax, shrink=0.85, label=r"$R_{max}/R_0$")

    plt.tight_layout()
    plt.show()

    print("done")

    # Global color normalization
    vmin = float(jnp.nanmin(cavit_heats))
    vmax = float(jnp.nanmax(cavit_heats))
    cmap = plt.get_cmap("inferno", 256)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Face grids (ny+1, nx+1)
    x_edges = edges_from_centers(x_centers)
    y_edges = edges_from_centers(y_centers)
    Xe, Ye = jnp.meshgrid(x_edges, y_edges, indexing="xy")

    # Convert to NumPy for Matplotlib
    Xe = np.asarray(Xe)
    Ye = np.asarray(Ye)
    cavit_heats_np = np.asarray(cavit_heats)
    z_centers_np = np.asarray(z_centers)

    # Create figure/axis
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each z plane
    for k, z0 in enumerate(z_centers_np):
        facecolors = cmap(norm(cavit_heats_np[k]))  # (ny, nx, 4)
        Ze = np.full_like(Xe, float(k), dtype=float)

        ax.plot_surface(
            Xe,
            Ye,
            Ze,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            edgecolor=(0, 0, 0, 0),  # fully transparent edge
            antialiased=False,
            shade=False,
            alpha=0.8,
        )

    # Aesthetics (cubic aspect, ticks at centers)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xticks(np.asarray(x_centers))
    ax.set_yticks(np.asarray(y_centers))

    if z_labels:
        ax.set_zticks(z_discrete_ticks)
        ax.set_zticklabels(z_labels)
    else:
        ax.set_zticks(np.linspace(1, len(z_centers), len(z_centers)))

    ax.set_xticks(np.linspace(x_centers.min(), x_centers.max(), 5))
    ax.set_yticks(np.linspace(y_centers.min(), y_centers.max(), 5))
    ax.grid(True, color="red", linewidth=0.8, linestyle="-")

    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (1, 0, 0, 1)  # red
        axis._axinfo["grid"]["linewidth"] = 0.1
        axis._axinfo["grid"]["linestyle"] = "-"

    plt.tight_layout()
    plt.show()


def freq_r0_chi():
    freq_values = jnp.linspace(0.1, 1.5, 30)
    r0_values = jnp.linspace(1.0, 5.0, 30)
    chi_values = jnp.linspace(0.05, 0.5, 5)

    def freq_r0_chi_kernel(freq, r0, chi):
        bubble = Bubble(R0=r0, chi=chi)
        pulse = Pulse(
            freq=freq,
            pressure=100e3,
            shape=shapes.Sine(),
            cycle_num=5,
            initial_time=1e-6,
        )
        return run_simulation_jit(
            bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6
        )

    heats, xg, yg, cavit_heats = run_3d_sweep(
        freq_values * 1e6,
        r0_values * 1e-6,
        chi_values,
        freq_r0_chi_kernel,
        x_name="Frequency",
        y_name="R0",
        z_name="Chi",
    )

    # Plot as stacked planes
    plot_stacked_planes(
        x_centers=freq_values,
        y_centers=r0_values,
        z_centers=chi_values,
        heats=heats,
        cavit_heats=cavit_heats,
        title=r"$(f, R_0, \chi)$",
        x_label="Frequency (MHz)",
        y_label="Initial Radius $R_0$ (µm)",
        z_label=r"Shell Elasticity $\chi$ (N/m)",
    )


def freq_r0_kappa():
    freq_values = jnp.linspace(0.1, 1.5, 50)
    r0_values = jnp.linspace(1.0, 5.0, 50)
    kappa_values = jnp.linspace(1e-9, 10e-9, 5)

    def freq_r0_kappa_kernel(freq, r0, kappa):
        bubble = Bubble(R0=r0, kappa_s=kappa, chi=0.38)
        pulse = Pulse(
            freq=freq,
            pressure=100e3,
            shape=shapes.Sine(),
            cycle_num=5,
            initial_time=1e-6,
        )
        return run_simulation_jit(
            bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6
        )

    heats, xg, yg, cavit_heats = run_3d_sweep(
        freq_values * 1e6,
        r0_values * 1e-6,
        kappa_values,
        freq_r0_kappa_kernel,
        x_name="Frequency",
        y_name="R0",
        z_name="Kappa",
    )

    # Plot as stacked planes
    plot_stacked_planes(
        x_centers=freq_values,
        y_centers=r0_values,
        z_centers=kappa_values,
        heats=heats,
        cavit_heats=cavit_heats,
        title=r"$(f, R_0, \kappa)$",
        x_label="Frequency (MHz)",
        y_label="Initial Radius $R_0$ (µm)",
        z_label=r"Surface Dilational Viscosity $\kappa$ (N)",
    )


def freq_chi_kappa():

    freq_values = jnp.linspace(0.1, 1.5, 50)
    chi_values = jnp.linspace(0.05, 0.5, 50)
    kappa_values = jnp.linspace(1e-9, 10e-9, 5)

    def freq_chi_kappa_kernel(freq, chi, kappa):
        bubble = Bubble(R0=4e-6, kappa_s=kappa, chi=chi)
        pulse = Pulse(
            freq=freq,
            pressure=100e3,
            shape=shapes.Sine(),
            cycle_num=5,
            initial_time=1e-6,
        )
        return run_simulation_jit(
            bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6
        )

    heats, xg, yg, cavit_heats = run_3d_sweep(
        freq_values * 1e6,
        chi_values,
        kappa_values,
        freq_chi_kappa_kernel,
        x_name="Frequency",
        y_name="Chi",
        z_name="Kappa",
    )

    # Plot as stacked planes
    plot_stacked_planes(
        x_centers=freq_values,
        y_centers=chi_values,
        z_centers=kappa_values,
        heats=heats,
        cavit_heats=cavit_heats,
        title=r"$(f, \chi, \kappa)$",
        x_label="Frequency (MHz)",
        y_label="Compression Modulus $\chi$ (N/m)",
        z_label=r"Surface Dilational Viscosity $\kappa$ (N)",
    )


def freq_r0_p():

    freq_values = jnp.linspace(0.1, 1.5, 100)
    r0_values = jnp.linspace(1.0, 5.0, 100)
    pressure_values = jnp.linspace(50, 150, 5)

    def freq_r0_p_kernel(freq, r0, p):
        bubble = Bubble(R0=r0)
        pulse = Pulse(
            freq=freq, pressure=p, shape=shapes.Sine(), cycle_num=5, initial_time=1e-6
        )
        return run_simulation_jit(
            bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6
        )

    heats, xg, yg, cavit_heats = run_3d_sweep(
        freq_values * 1e6,
        r0_values * 1e-6,
        pressure_values * 1e3,
        freq_r0_p_kernel,
        x_name="Frequency",
        y_name="R0",
        z_name="Pressure",
    )

    # Plot as stacked planes
    plot_stacked_planes(
        x_centers=freq_values,
        y_centers=r0_values,
        z_centers=pressure_values,
        heats=heats,
        cavit_heats=cavit_heats,
        title=r"$(f, R_0, P)$",
        x_label="Frequency (MHz)",
        y_label="Initial Radius $R_0$ (µm)",
        z_label="Pressure (kPa)",
    )


def freq_r0_gamma():
    freq_values = jnp.linspace(0.1, 1.5, 100)
    r0_values = jnp.linspace(1.0, 5.0, 100)
    gamma_values = jnp.linspace(1.05, 2, 5)

    def freq_r0_gamma_kernel(freq, r0, gamma):
        bubble = Bubble(R0=r0, gamma=gamma)
        pulse = Pulse(
            freq=freq,
            pressure=100e3,
            shape=shapes.Sine(),
            cycle_num=5,
            initial_time=1e-6,
        )
        return run_simulation_jit(
            bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6
        )

    heats, xg, yg, cavit_heats = run_3d_sweep(
        freq_values * 1e6,
        r0_values * 1e-6,
        gamma_values,
        freq_r0_gamma_kernel,
        x_name="Frequency",
        y_name="R0",
        z_name="Gas Constant",
    )

    # Plot as stacked planes
    plot_stacked_planes(
        x_centers=freq_values,
        y_centers=r0_values,
        z_centers=gamma_values,
        heats=heats,
        cavit_heats=cavit_heats,
        title=r"$(f, R_0, \gamma)$",
        x_label="Frequency (MHz)",
        y_label="Initial Radius $R_0$ (µm)",
        z_label="Gas constant $\gamma$",
    )


def freq_r0_shapes():
    freq_values = jnp.linspace(0.1, 1.5, 50)
    r0_values = jnp.linspace(1.0, 5.0, 50)
    shape_types = [
        shapes.Sine,
        shapes.Sawtooth,
        shapes.InvertedSawtooth,
        shapes.Square,
        shapes.Triangle,
    ]
    z_vals = jnp.linspace(1, 5, 5)

    def freq_r0_shapes_kernel(freq, r0, shape_type):
        bubble = Bubble(R0=r0)
        pulse = Pulse(
            freq=freq,
            pressure=100e3,
            shape=shape_type(),
            cycle_num=5,
            initial_time=1e-6,
        )
        return run_simulation_jit(
            bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6
        )

    heats, xg, yg, cavit_heats = run_3d_sweep(
        freq_values * 1e6,
        r0_values * 1e-6,
        shape_types,
        freq_r0_shapes_kernel,
        x_name="Frequency",
        y_name="R0",
        z_name="Shape Type",
    )

    # Plot as stacked planes
    plot_stacked_planes(
        x_centers=freq_values,
        y_centers=r0_values,
        z_centers=shape_types,
        heats=heats,
        cavit_heats=cavit_heats,
        title=r"$(f, R_0$, Shapes)",
        x_label="Frequency (MHz)",
        y_label="Initial Radius $R_0$ (µm)",
        z_label="Shape",
    )


def freq_r0_rectangle():
    freq_values = jnp.linspace(0.1, 1.5, 50)
    r0_values = jnp.linspace(1.0, 5.0, 50)
    shape_types = [
        shapes.Rect95,
        shapes.Rect75NegPos,
        shapes.Rect25NegPos,
        shapes.SquareNegPos,
    ]
    z_vals = jnp.linspace(1, 4, 4)

    def freq_r0_shapes_kernel(freq, r0, shape_type):
        bubble = Bubble(R0=r0)
        pulse = Pulse(
            freq=freq,
            pressure=100e3,
            shape=shape_type(),
            cycle_num=5,
            initial_time=1e-6,
        )
        return run_simulation_jit(
            bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6
        )

    heats, xg, yg, cavit_heats = run_3d_sweep(
        freq_values * 1e6,
        r0_values * 1e-6,
        shape_types,
        freq_r0_shapes_kernel,
        x_name="Frequency",
        y_name="R0",
        z_name="Shape Type",
    )

    # Plot as stacked planes
    plot_stacked_planes(
        x_centers=freq_values,
        y_centers=r0_values,
        z_centers=shape_types,
        heats=heats,
        cavit_heats=cavit_heats,
        title=r"$(f, R_0$, Shapes)",
        x_label="Frequency (MHz)",
        y_label="Initial Radius $R_0$ (µm)",
        z_label="Shape",
    )


def freq_r0_square_posneg():
    freq_values = jnp.linspace(0.1, 1.5, 50)
    r0_values = jnp.linspace(1.0, 5.0, 50)
    shape_types = [shapes.Square, shapes.SquareNegPos]
    z_vals = jnp.linspace(1, 2, 2)

    def freq_r0_shapes_kernel(freq, r0, shape_type):
        bubble = Bubble(R0=r0)
        pulse = Pulse(
            freq=freq,
            pressure=100e3,
            shape=shape_type(),
            cycle_num=5,
            initial_time=1e-6,
        )
        return run_simulation_jit(
            bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6
        )

    heats, xg, yg, cavit_heats = run_3d_sweep(
        freq_values * 1e6,
        r0_values * 1e-6,
        shape_types,
        freq_r0_shapes_kernel,
        x_name="Frequency",
        y_name="R0",
        z_name="Shape Type",
    )

    # Plot as stacked planes
    plot_stacked_planes(
        x_centers=freq_values,
        y_centers=r0_values,
        z_centers=shape_types,
        heats=heats,
        cavit_heats=cavit_heats,
        title=r"$(f, R_0$, Shapes)",
        x_label="Frequency (MHz)",
        y_label="Initial Radius $R_0$ (µm)",
        z_label="Shape",
    )


# STILL NEED

if __name__ == "__main__":
    freq_r0_chi()
