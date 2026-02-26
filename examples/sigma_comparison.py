#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for comparing Marmottant and MarmottantGompertz bubble models.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jbubble.bubble import MarmottantGompertz, Marmottant, RayleighPlesset


def plot_surface_tension_comparison(bubbles, labels=None, R_min=None, R_max=None, num_points=500):
    """
    Plot and compare surface tension vs radius for multiple bubble models.

    Parameters
    ----------
    bubbles : list
        List of bubble model instances
    labels : list, optional
        Labels for each bubble model (defaults to class names)
    R_min : float, optional
        Minimum radius to plot (defaults to 0.5*R0)
    R_max : float, optional
        Maximum radius to plot (defaults to 2*R0)
    num_points : int
        Number of points to sample between R_min and R_max
    """
    if labels is None:
        labels = [bubble.__class__.__name__ for bubble in bubbles]
    
    # Use the first bubble's R0 for range
    R0 = bubbles[0].R0
    R_min = R_min or 0.5 * R0
    R_max = R_max or 1.5 * R0

    R_vals = jnp.linspace(R_min, R_max, num_points)

    plt.figure(figsize=(10, 6))
    
    # Plot all bubbles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (bubble, label) in enumerate(zip(bubbles, labels)):
        sigma_vals = bubble.surface_tension(R_vals)
        plt.plot(R_vals * 1e6, sigma_vals * 1e3, lw=2.5, label=label, color=colors[i % len(colors)])
    
    plt.xlabel("Radius [µm]", fontsize=12)
    plt.ylabel("Surface tension [mN/m]", fontsize=12)
    plt.title("Surface Tension Comparison: Marmottant vs Marmottant-Gompertz", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create bubble models with R0 = 4 µm
    bubble_marmottant = Marmottant(R0=4e-6, chi = 0.5)  # R0 = 4 µm
    bubble_marmottant_gompertz = MarmottantGompertz(R0=4e-6, chi = 0.5)  # R0 = 4 µm
    bubble_rayleigh_plesset = RayleighPlesset(R0=4e-6)  # R0 = 4 µm
    print("=" * 70)
    print("MARMOTTANT BUBBLE MODEL")
    print("=" * 70)
    print(bubble_marmottant)
    
    print("\n" + "=" * 70)
    print("MARMOTTANT-GOMPERTZ BUBBLE MODEL")
    print("=" * 70)
    print(bubble_marmottant_gompertz)
    
    print("\n" + "=" * 70)
    print("RAYLEIGH-PLESSET BUBBLE MODEL")
    print("=" * 70)
    print(bubble_rayleigh_plesset)

    print("\n" + "=" * 70)
    print("PLOTTING SURFACE TENSION COMPARISON")
    print("=" * 70)
    plot_surface_tension_comparison(
        [bubble_marmottant, bubble_marmottant_gompertz, bubble_rayleigh_plesset],
        labels=["Marmottant", "Marmottant-Gompertz", "Rayleigh-Plesset"]
    )