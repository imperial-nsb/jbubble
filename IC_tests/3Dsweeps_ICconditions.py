#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized Inertial Cavitation Sweep with labeled axes
Side-by-side plotting for absolute and binary heatmaps
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
import jbubble.IC_condition as condition

from jax import config
config.update("jax_enable_x64", True)

units = Units()
save_spec = SaveSpec(num_samples=1000)
run_simulation_jit = jax.jit(run_simulation)
Bubble = MarmottantGompertz

# ---------------- 2D Sweep ----------------
def run_2d_sweep(x_values, y_values, kernel_func):
    x_grid, y_grid = jnp.meshgrid(x_values, y_values)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()

    vectorized_run = jax.vmap(kernel_func)
    jit_vectorized_run = jax.jit(vectorized_run)
    results_flat = jit_vectorized_run(x_flat, y_flat)
    _ = results_flat.radius.block_until_ready()

    results_grid = jax.tree.map(
        lambda x: x.reshape(*x_grid.shape, *x.shape[1:]),
        results_flat
    )
    return results_grid, x_grid, y_grid

# ---------------- 3D Sweep for multiple thresholds ----------------
def run_3d_sweep_multi_threshold(x_values, y_values, z_values, kernel3d,
                                 expansion_thresh=condition.expansion_thresholds,
                                 KE_thresh=condition.KE_thresholds):
    heats_list = []
    cavit_dict = {key: [] for key in expansion_thresh.keys()}
    cavit_dict.update({key: [] for key in KE_thresh.keys()})
    metric_dict = {key: [] for key in expansion_thresh.keys()}
    metric_dict.update({key: [] for key in KE_thresh.keys()})

    for z0 in z_values:
        def kernel2d(x, y):
            return kernel3d(x, y, z0)

        res, x_grid, y_grid = run_2d_sweep(x_values, y_values, kernel2d)

        heat2d = res.radius.max(axis=-1) / res.bubble.R0
        heats_list.append(heat2d)

        # --- expansion thresholds ---
        for key, func in expansion_thresh.items():
            flags = jnp.zeros_like(heat2d)
            metrics = jnp.zeros_like(heat2d)
            for i in range(heat2d.shape[0]):
                for j in range(heat2d.shape[1]):
                    val, flag = func(res.radius[i,j,:], res.bubble.R0[i,j])
                    flags = flags.at[i,j].set(flag)
                    metrics = metrics.at[i,j].set(val)
            cavit_dict[key].append(flags)
            metric_dict[key].append(metrics)

        rho_L = res.bubble.rho_L[0,0] 
        c_L = res.bubble.c_L[0,0] 
        # --- KE thresholds ---
        for key, func in KE_thresh.items():
            flags = jnp.zeros_like(heat2d)
            metrics = jnp.zeros_like(heat2d)
            for i in range(heat2d.shape[0]):
                for j in range(heat2d.shape[1]):
                    val, flag = func(res.radius[i,j,:], res.radial_velocity[i,j,:],
                                     res.bubble.R0[i,j], rho_L, c_L)
                    flags = flags.at[i,j].set(flag)
                    metrics = metrics.at[i,j].set(val)
            cavit_dict[key].append(flags)
            metric_dict[key].append(metrics)

    heats = jnp.stack(heats_list, axis=0)
    for key in cavit_dict:
        cavit_dict[key] = jnp.stack(cavit_dict[key], axis=0)
        metric_dict[key] = jnp.stack(metric_dict[key], axis=0)

    return metric_dict, cavit_dict, x_grid, y_grid

# ---------------- Plotting ----------------
def plot_thresholds_side_by_side(x_centers, y_centers, z_centers, metric_dict, cavit_dict,
                                 title_prefix="Cavitation Threshold",
                                 x_label="X", y_label="Y", z_label="Z"):
    def edges_from_centers(centers):
        centers = np.asarray(centers)
        midpoints = 0.5 * (centers[:-1] + centers[1:])
        left_edge = centers[0] - (midpoints[0] - centers[0])
        right_edge = centers[-1] + (centers[-1] - midpoints[-1])
        return np.concatenate(([left_edge], midpoints, [right_edge]), axis=0)

    x_edges = edges_from_centers(x_centers)
    y_edges = edges_from_centers(y_centers)
    Xe, Ye = np.meshgrid(x_edges, y_edges, indexing='xy')
    z_centers_np = np.asarray(z_centers)

    for key in metric_dict.keys():
        metric_np = np.asarray(metric_dict[key])
        cavit_np = np.asarray(cavit_dict[key])

        fig, axes = plt.subplots(1, 2, figsize=(14,6), subplot_kw={'projection': '3d'})

        # --- metric heatmap ---
        vmin, vmax = float(np.nanmin(metric_np)), float(np.nanmax(metric_np))
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap('turbo', 256)

        ax = axes[0]
        for k in range(len(z_centers_np)):
            facecolors = cmap(norm(metric_np[k]))
            Ze = np.full_like(Xe, float(k))
            ax.plot_surface(Xe, Ye, Ze, facecolors=facecolors,
                            linewidth=0,            
            edgecolor='none',       
            antialiased=False,      
            shade=False,
            alpha=0.8  )
        ax.set_title(f"{title_prefix}: {key}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.set_xticks(np.linspace(x_centers.min(), x_centers.max(), 5))
        ax.set_yticks(np.linspace(y_centers.min(), y_centers.max(), 5))
        ax.set_zticks(np.arange(len(z_centers_np)))
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6,
                     label='Rmax/R0' if 'expansion' in key else 'Normalized KE')

        # --- Binary heatmap ---
        ax = axes[1]
        for k in range(len(z_centers_np)):
            facecolors = np.where(cavit_np[k], 'red', 'white')
            Ze = np.full_like(Xe, float(k))
            ax.plot_surface(Xe, Ye, Ze, facecolors=facecolors,
            linewidth=0,            
            edgecolor='none',       
            antialiased=False,      
            shade=False,
            alpha=0.8  )
        ax.set_title(f"{title_prefix}: {key} (Binary)")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.set_xticks(np.linspace(x_centers.min(), x_centers.max(), 5))
        ax.set_yticks(np.linspace(y_centers.min(), y_centers.max(), 5))
        ax.set_zticks(np.arange(len(z_centers_np)))

        plt.tight_layout()
        plt.show()

# ---------------- Example kernel ----------------
def freq_r0_chi_kernel(freq, r0, chi):
    bubble = Bubble(R0=r0, chi=chi)
    pulse = Pulse(freq=freq, pressure=100e3, shape=shapes.Triangle(),
                  cycle_num=5, initial_time=1e-6)
    return run_simulation_jit(bubble, pulse, units=units, save_spec=save_spec, window_s=20e-6)

# ---------------- Main ----------------
if __name__ == "__main__":
    freq_values = jnp.linspace(0.1, 1.5, 30)   # MHz
    r0_values   = jnp.linspace(1.0, 5.0, 30)   # µm
    chi_values  = jnp.linspace(0.05, 0.5, 5)   # N/m

    metric_dict, cavit_dict, xg, yg = run_3d_sweep_multi_threshold(
        freq_values*1e6, r0_values*1e-6, chi_values, freq_r0_chi_kernel
    )

    plot_thresholds_side_by_side(
        x_centers=freq_values,
        y_centers=r0_values,
        z_centers=chi_values,
        metric_dict=metric_dict,
        cavit_dict=cavit_dict,
        title_prefix="Inertial Cavitation",
        x_label="Frequency (MHz)",
        y_label="Initial Radius R0 (µm)",
        z_label="Shell Elasticity χ (N/m)"
    )