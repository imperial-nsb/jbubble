import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jbubble.bubble.marmottant_gompertz import MarmottantGompertz
from jbubble.units import Units
from jbubble.pulse import Pulse
from jbubble.solver import SaveSpec
from jbubble.simulation import run_simulation
from jbubble.shapes import Sine, Triangle, Square, Quadratic

units = Units()
save_spec = SaveSpec(num_samples=1000)

# ==========================================================
# Configuration
# ==========================================================

bubble = MarmottantGompertz(
    R0=3e-6,
    chi=0.38,
    kappa_s=3e-9
)

pressures = jnp.linspace(10e3, 150e3, 50)
shape_classes = [Sine, Triangle, Square, Quadratic]

# ==========================================================
# Create a JAX-friendly kernel for a single shape
# ==========================================================

def make_pressure_kernel(shape_cls):
    pulse_shape = shape_cls()

    def single_pressure_kernel(pressure):
        pulse = Pulse(
            shape=pulse_shape,
            freq=300e3,
            pressure=pressure,
            cycle_num=4,
            initial_time=1e-6,
            apply_hann=False,
        )

        result = run_simulation(
            bubble=bubble,
            pulse=pulse,
            units=units,
            save_spec=save_spec,
            window_s=20e-6,
            dt0=1e-3,
            max_steps=10_000,
            progress=False,
        )

        R = result.radius
        Rdot = result.radial_velocity

        E = 2.0 * jnp.pi * bubble.rho_L * R**3 * Rdot**2
        E_max = jnp.max(E)
        E_norm = E_max / (bubble.rho_L * bubble.R0**3 * bubble.c_L**2)

        Rmax_norm = jnp.max(R) / bubble.R0

        return jnp.array([E_norm, Rmax_norm], dtype=jnp.float32)

    # Vectorize over pressures
    return jax.vmap(single_pressure_kernel, in_axes=(0,))

# ==========================================================
# Run sweep for all shapes (no Python loop over pressures)
# ==========================================================

print("Running fully vectorized sweep...")
start_time = time.time()

all_results = []

for shape_cls in shape_classes:
    pressure_kernel = make_pressure_kernel(shape_cls)
    results = pressure_kernel(pressures)  # shape: (num_pressures, 2)
    all_results.append(results)

all_results = jnp.stack(all_results, axis=1)  # shape: (num_pressures, num_shapes, 2)

KE_grid = all_results[:, :, 0]
Rmax_grid = all_results[:, :, 1]

end_time = time.time()
print(f"Sweep completed in {end_time - start_time:.2f} seconds")

# ==========================================================
# Convert to NumPy for plotting
# ==========================================================

KE_np = np.array(KE_grid)
Rmax_np = np.array(Rmax_grid)

# ==========================================================
# Plot side by side
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1, ax2 = axes

for i, shape_cls in enumerate(shape_classes):
    ax1.plot(pressures, KE_np[:, i]*100, 'o-', linewidth=2, label=shape_cls.__name__)
ax1.set_xlabel("Driving Pressure (Pa)")
ax1.set_ylabel("Normalized Peak Kinetic Energy %")
ax1.set_title("Normalized Peak Kinetic Energy vs Pressure")
ax1.grid(True, alpha=0.3)
ax1.legend()

for i, shape_cls in enumerate(shape_classes):
    ax2.plot(pressures, Rmax_np[:, i], 'o-', linewidth=2, label=shape_cls.__name__)
ax2.set_xlabel("Driving Pressure (Pa)")
ax2.set_ylabel("R_max / R0")
ax2.set_title("Maximum Bubble Expansion vs Pressure")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# Automatic inertial threshold detection
# ==========================================================

d2 = jnp.gradient(jnp.gradient(KE_grid, axis=0), axis=0)
threshold_indices = jnp.argmax(d2, axis=0)
threshold_pressures = pressures[threshold_indices]

print("\nEstimated Inertial Thresholds:")
for i, shape_cls in enumerate(shape_classes):
    print(f"{shape_cls.__name__}: {float(threshold_pressures[i]/1e3):.1f} kPa")

    import matplotlib.pyplot as plt
import jax.numpy as jnp

