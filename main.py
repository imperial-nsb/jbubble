import jax
import matplotlib.pyplot as plt
import numpy as np
import time
from chex import assert_max_traces

from jbubble import (
    Units,
    SaveSpec,
    compute_radius_metrics,
    default_bubble,
    default_pulse,
    run_simulation,
)


def demo_run(R0=4e-6, pressure=500e3, freq=800e3):
    units = Units()
    bubbleA = default_bubble(R0=R0)
    # bubbleB = default_bubble(R0=R0)

    pulseA = default_pulse(freq=freq, pressure=pressure)
    pulseB = default_pulse(freq=freq * 1.1, pressure=pressure * 1.1)
    pulseC = default_pulse(freq=freq * 1.1, pressure=pressure * 1.1)

    save_spec = SaveSpec(num_samples=1000)

    jit_time = time.perf_counter()
    jit_sim = jax.jit(assert_max_traces(run_simulation, n=1))
    end_jit_time = time.perf_counter()
    print(f"JIT compilation took {end_jit_time - jit_time:.4f} seconds")

    start_A = time.perf_counter()
    resultA = jit_sim(bubble=bubbleA, pulse=pulseA, units=units, save_spec=save_spec)
    end_A = time.perf_counter()

    start_B = time.perf_counter()
    resultB = jit_sim(bubble=bubbleA, pulse=pulseB, units=units, save_spec=save_spec)
    end_B = time.perf_counter()

    start_C = time.perf_counter()
    resultC = jit_sim(bubble=bubbleA, pulse=pulseC, units=units, save_spec=save_spec)
    end_C = time.perf_counter()

    print(f"Simulation A took {end_A - start_A:.4f} seconds")
    print(f"Simulation B took {end_B - start_B:.4f} seconds")
    print(f"Simulation C took {end_C - start_C:.4f} seconds")
    metricsA = compute_radius_metrics(resultA)
    metricsB = compute_radius_metrics(resultB)

    ts_us_A = np.array(resultA.ts) / units.T_scale
    driving_kpa_A = np.array(resultA.driving_pressure) / units.P_scale
    radius_um_A = np.array(resultA.radius) / units.L_scale
    ts_us_B = np.array(resultB.ts) / units.T_scale
    driving_kpa_B = np.array(resultB.driving_pressure) / units.P_scale
    radius_um_B = np.array(resultB.radius) / units.L_scale

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.set_ylabel("Driving Pressure (kPa)", color="tab:blue")
    ax1.plot(ts_us_A, driving_kpa_A, label="Driving Pressure A", color="tab:blue")
    ax1.plot(ts_us_B, driving_kpa_B, label="Driving Pressure B", color="tab:cyan")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.set_xlabel("Time (μs)")
    ax2.set_ylabel("Radius (μm)", color="tab:orange")
    ax2.plot(ts_us_A, radius_um_A, label="Radius A / Time", color="tab:orange")
    ax2.plot(ts_us_B, radius_um_B, label="Radius B / Time", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    print("Converged A:", resultA.converged)
    print("Converged B:", resultB.converged)
    for key, value in metricsA.items():
        print(f"{key} A: {value:.3e}")
    for key, value in metricsB.items():
        print(f"{key} B: {value:.3e}")


def main():
    demo_run()


if __name__ == "__main__":
    main()
