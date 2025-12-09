import matplotlib.pyplot as plt

from jbubble import (
    Units,
    SaveSpec,
    arrays_from_result,
    compute_radius_metrics,
    run_simulation,
)
from jbubble.bubble import Bubble
from jbubble.simulation import build_pulse
import time
from jax import jit


def demo():
    units = Units()
    bubbleA = Bubble(R0=2e-6)
    bubbleB = Bubble(R0=3e-6)
    freq = 300e3
    pressure = 100e3
    pulse = build_pulse(
        "sine",
        freq=freq,
        pressure=pressure,
        cycle_num=10,
        initial_time=1e-6,
        apply_hann=False,
    )

    save_spec = SaveSpec(num_samples=1000)

    # JIT compile the simulation function
    jit_run_simulation = jit(run_simulation)

    # First run (includes compilation time)
    startA = time.perf_counter()
    resultA = jit_run_simulation(
        bubble=bubbleA,
        pulse=pulse,
        units=units,
        save_spec=save_spec,
    )
    endA = time.perf_counter()
    print(f"First run (with JIT compilation): {endA - startA:.4f} seconds")

    # Second run (just execution time)
    startB = time.perf_counter()
    resultB = jit_run_simulation(
        bubble=bubbleB,
        pulse=pulse,
        units=units,
        save_spec=save_spec,
    )
    endB = time.perf_counter()
    print(f"Second run (JIT compiled): {endB - startB:.4f} seconds")

    metricsA = compute_radius_metrics(resultA)
    arraysA = arrays_from_result(resultA)
    radius_umA = arraysA.radius_um

    metricsB = compute_radius_metrics(resultB)
    arraysB = arrays_from_result(resultB)
    radius_umB = arraysB.radius_um

    ts_us = arraysA.time_us
    driving_kpa = arraysA.pressure_kpa

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.set_ylabel("Driving Pressure (kPa)", color="black")
    ax1.plot(ts_us, driving_kpa, label="Driving Pressure", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.set_xlabel("Time (μs)")
    ax2.set_ylabel("Radius (μm)", color="black")
    ax2.plot(ts_us, radius_umA, label="Radius / Time A", color="tab:blue")
    ax2.plot(ts_us, radius_umB, label="Radius / Time B", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    print("Converged A:", resultA.converged)
    print("Metrics A:")
    for key, value in metricsA.items():
        print(f"{key}: {value:.3e}")
    print("Converged B:", resultB.converged)
    print("Metrics B:")
    for key, value in metricsB.items():
        print(f"{key}: {value:.3e}")


if __name__ == "__main__":
    demo()
