import matplotlib.pyplot as plt
from jbubble import (
    SaveSpec,
    Pulse,
    run_simulation,
)
from jbubble.bubble import Marmottant
import jbubble.shapes as shapes


bubble = Marmottant(R0=3.0e-6)
save_spec = SaveSpec(num_samples=1000)

pulse_A = Pulse(
    freq=500e3,
    pressure=25e3,
    shape=shapes.Sine(),
    cycle_num=5,
    initial_time=1e-6,
    apply_hann=False,
)

pulse_B = Pulse(
    freq=500e3,
    pressure=100e3,
    shape=shapes.Sine(),
    cycle_num=5,
    initial_time=1e-6,
    apply_hann=False,
)

result_A = run_simulation(
    bubble=bubble,
    pulse=pulse_A,
    save_spec=save_spec,
)

result_B = run_simulation(
    bubble=bubble,
    pulse=pulse_B,
    save_spec=save_spec,
)

# ---
time_axis_A = result_A.ts  # [s]
pressure_input_A = pulse_A(time_axis_A)  # [Pa]
radius_output_A = result_A.radius  # [m]

time_axis_B = result_B.ts  # [s]
pressure_input_B = pulse_B(time_axis_B)  # [Pa]
radius_output_B = result_B.radius  # [m]

# --- PLOT A ---
# Plot the results with shared time axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top plot: Driving Pressure
ax1.set_title("Bubble Radius vs Time (Stable Regime)")
ax1.plot(time_axis_A * 1e6, pressure_input_A / 1e3, color="green")
ax1.set_ylabel("Driving Pressure (kPa)")
ax1.set_ylim(-220, 220)
ax1.grid(True)

# Bottom plot: Radius
ax2.plot(time_axis_A * 1e6, radius_output_A * 1e6, color="blue")
ax2.set_ylabel("Radius (µm)")
ax2.set_xlabel("Time (µs)")
ax2.set_ylim(0.7, 6.2)
ax2.grid(True)

plt.tight_layout()
plt.show()

# --- PLOT B ---
# Plot the results with shared time axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top plot: Driving Pressure
ax1.set_title("Bubble Radius vs Time (Inertial Regime)")
ax1.plot(time_axis_B * 1e6, pressure_input_B / 1e3, color="green")
ax1.set_ylabel("Driving Pressure (kPa)")
ax1.set_ylim(-220, 220)
ax1.grid(True)

# Bottom plot: Radius
ax2.plot(time_axis_B * 1e6, radius_output_B * 1e6, color="blue")
ax2.set_ylabel("Radius (µm)")
ax2.set_xlabel("Time (µs)")
ax2.set_ylim(0.7, 6.2)
ax2.grid(True)

plt.tight_layout()
plt.show()
# ---
