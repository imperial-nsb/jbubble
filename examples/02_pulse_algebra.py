"""02 Pulse Algebra

Demonstrates the composable pulse system.

jbubble allows you create complex driving waveforms by adding, 
scaling, and windowing simpler pulse primitives.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jbubble.pulse import ToneBurst, HannEnvelope, Sine

# 1. Create two basic pulses
# A 1 MHz burst (standard)
p1 = ToneBurst(
    freq=1e6, 
    pressure=50e3, 
    shape=Sine(), 
    cycle_num=5
)

# A 2.5 MHz burst, delayed by 6 microseconds
p2 = ToneBurst(
    freq=2.5e6, 
    pressure=30e3, 
    shape=Sine(), 
    cycle_num=15, 
    initial_time=6e-6
)

# 2. Combine them using Pulse Algebra
# Addition: creates superpositions
# Multiplication: scales the amplitude
# .windowed(): applies a new envelope to the whole assembly
combined = (p1 + p2) * 0.7 
final_pulse = combined.windowed(HannEnvelope())

# 3. Evaluate the pulse over a time range
# Since Pulses are JAX-compatible, we can use vmap for fast evaluation.
ts = jnp.linspace(0, 15e-6, 1000)
pressures = jax.vmap(final_pulse)(ts)

# 4. Plot the result
plt.figure(figsize=(10, 4))
plt.plot(ts * 1e6, pressures / 1e3, lw=2, color="crimson")
plt.fill_between(ts * 1e6, pressures / 1e3, alpha=0.1, color="crimson")

plt.xlabel("Time (µs)")
plt.ylabel("Acoustic Pressure (kPa)")
plt.title("Pulse Algebra: (1MHz + Delayed 2.5MHz) * 0.7 [Hann Windowed]")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Total pulse duration: {final_pulse.duration * 1e6:.2f} µs")
