"""06 JIT Timing Demo

Demonstrates the performance benefits of JAX's Just-In-Time (JIT) compilation.

In bubble dynamics, we often need to run the same physics model
many times with different driving pulses. jbubble leverages
JAX's XLA compiler to turn your Python simulation code into
highly optimized machine code.

This example measures:
1. The time for the first run (includes compilation)
2. The time for subsequent runs (pure execution)
"""

import time
import jax
import matplotlib.pyplot as plt
from jbubble import run_simulation
from jbubble.bubble.eom import RayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.solver import SaveSpec

# 1. Setup physics
eom = RayleighPlesset(
    gas=PolytropicGas(gamma=1.4),
    shell=NoShell(sigma=0.072),
    medium=NewtonianMedium(mu=0.001),
    R0=2e-6,
    P_amb=101325.0,
    rho_L=998.0,
)

# 2. Define the simulation function and JIT compile it
# This function is what gets compiled to XLA.
jit_sim = jax.jit(run_simulation)


def run_timed_demo(pressure):
    pulse = ToneBurst(freq=1e6, pressure=pressure, shape=Sine(), cycle_num=5)

    start = time.perf_counter()
    # Note: .radius.block_until_ready() is important for accurate timing
    # because JAX is asynchronous.
    result = jit_sim(eom, pulse, save_spec=SaveSpec(num_samples=1000))
    result.radius.block_until_ready()
    end = time.perf_counter()

    return result, end - start


# 3. Execute runs
pressures = [100e3, 150e3, 200e3, 250e3]
times = []

print("Running JIT timing demonstration...")

for i, p in enumerate(pressures):
    res, duration = run_timed_demo(p)
    times.append(duration)

    label = "FIRST RUN (includes JIT)" if i == 0 else f"RUN {i + 1} (cached)"
    print(f"{label:<25} : {duration:.4f} seconds")

# 4. Summary
speedup = times[0] / times[1]
print(f"\nSubsequent runs are ~{speedup:.1f}x faster than the first run.")

# 5. Simple plot of one result
plt.figure(figsize=(8, 4))
plt.plot(res.ts * 1e6, res.radius * 1e6, color="forestgreen", lw=2)
plt.xlabel("Time (µs)")
plt.ylabel("Radius (µm)")
plt.title(f"Simulation result for {pressures[-1] / 1e3} kPa")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
