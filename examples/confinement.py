#%%
"""
Test script for multi-model bubble simulations.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jbubble import shapes
from jbubble.bubble import SphericalConfinement
from jbubble.units import Units
from jbubble.pulse import Pulse
from jbubble.solver import SaveSpec
from jbubble.simulation import run_simulation
from matplotlib.animation import FuncAnimation


pulse = Pulse(
    shape=shapes.Square(),
    freq=300e3,
    pressure=10e3,
    cycle_num=20,
    initial_time=1e-6,
    apply_hann=False,
)

units = Units()
save_spec = SaveSpec(num_samples=1000)

result_spherical_confinement = run_simulation(
    bubble=SphericalConfinement(R0=4e-6, vessel_radius=10e-6),  # R0 = 4 µm
    pulse=pulse,
    units=units,
    save_spec=save_spec,
    window_s=20e-6,
    dt0=1e-3,
    max_steps=10_000,
    progress=True,
)

# Convert to microns and microseconds
t = result_spherical_confinement.ts * 1e6
bubble_r = result_spherical_confinement.radius  *1e6
vessel_r = result_spherical_confinement.vessel_radius * 1e6

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))

# Set limits dynamically
max_radius = np.max(vessel_r) * 1.1
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(-max_radius, max_radius)
ax.set_aspect('equal')
ax.set_title("Bubble & Vessel Oscillation")
ax.set_xlabel("µm")
ax.set_ylabel("µm")

# Create circle artists
bubble_circle = plt.Circle((0, 0), bubble_r[0], color='tab:blue', alpha=0.6, label="Bubble")
vessel_circle = plt.Circle((0, 0), vessel_r[0], fill=False, 
                           edgecolor='tab:red', linewidth=2.5, label="Vessel")

ax.add_patch(vessel_circle)
ax.add_patch(bubble_circle)
ax.legend()

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Animation update function
def update(frame):
    bubble_circle.set_radius(bubble_r[frame])
    vessel_circle.set_radius(vessel_r[frame])
    time_text.set_text(f"t = {t[frame]:.2f} µs")
    return bubble_circle, vessel_circle, time_text

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
#ani.save("bubble_vessel_animation.gif",writer="pillow",fps=30)
plt.show()

# ------------------------------------------------------------------
# GENERIC SWEEP FUNCTION
# ------------------------------------------------------------------
PRESSURE = 10e3  # 10 kPa
CYCLES = 20
INITIAL_TIME = 1e-6  # 1 microsecond    

def run_2d_sweep(x_values, y_values, kernel):

    x_grid, y_grid = jnp.meshgrid(x_values, y_values)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()

    vmapped = jax.vmap(kernel)
    jit_vmapped = jax.jit(vmapped)

    print(f"Running {x_flat.size} simulations...")
    t0 = time.time()

    results = jit_vmapped(x_flat, y_flat)
    _ = results.block_until_ready()

    print(f"Done in {time.time() - t0:.2f} seconds")

    results = results.reshape(x_grid.shape)

    return results, x_grid, y_grid


# ------------------------------------------------------------------
# MODEL KERNELS (ALL IDENTICAL PHYSICS)
# ------------------------------------------------------------------

def make_pulse(freq):
    return Pulse(
        freq=freq,
        pressure=PRESSURE,
        shape=shapes.Sine(),
        cycle_num=CYCLES,
        initial_time=INITIAL_TIME,
    )


def spherical_confinement(freq, r0):
    bubble = SphericalConfinement(R0=r0, vessel_radius=10e-6, vessel_E = 0.1e6)  # Fixed vessel radius for sweep
    pulse = make_pulse(freq)

    result = run_simulation(
        bubble, pulse,
        units=units,
        save_spec=save_spec,
        window_s=20e-6
    )

    return result.radius.max() / r0


# ------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------

def plot_heatmaps(freq_grid, r0_grid,
                  data):

    

    title = "Spherical Confinement"

    
    plt.pcolormesh(
            r0_grid * 1e6,
            freq_grid / 1e6,
            data,
            shading="auto",
            cmap="inferno",
        )
    plt.title(title)
    plt.ylabel("Frequency (MHz)")
    plt.xlabel("R₀ (µm)")
    plt.colorbar(label="Rmax / R0")


    plt.tight_layout()
    plt.show()



r0_values = jnp.linspace(1e-6, 5e-6, 10)
freq_values = jnp.linspace(0.1e6, 1.4e6, 10)


print("Sweep: Spherical Confinement")
data_spherical, freq_grid, r0_grid = run_2d_sweep(
    freq_values, r0_values, spherical_confinement
    )

plot_heatmaps(freq_grid, r0_grid,
                  data_spherical)



