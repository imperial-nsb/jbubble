import matplotlib.pyplot as plt

from jbubble import (
    Units,
    SaveSpec,
    arrays_from_result,
    compute_radius_metrics,
    default_bubble,
    run_simulation,
)
from jbubble.bubble import Bubble
from jbubble.simulation import build_pulse
import time
from jax import jit
import equinox as eqx


def demo():
    units = Units()
    bubble = default_bubble(R0=2e-6)
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
    jit_run_simulation = eqx.filter_jit(run_simulation)

    # First run (includes compilation time)
    start1 = time.perf_counter()
    result = jit_run_simulation(
        bubble=bubble,
        pulse=pulse,
        units=units,
        save_spec=save_spec,
    )
    end1 = time.perf_counter()
    print(f"First run (with JIT compilation): {end1 - start1:.4f} seconds")

    # Second run (just execution time)
    start2 = time.perf_counter()
    result = jit_run_simulation(
        bubble=bubble,
        pulse=pulse,
        units=units,
        save_spec=save_spec,
    )
    end2 = time.perf_counter()
    print(f"Second run (JIT compiled): {end2 - start2:.4f} seconds")


    metrics = compute_radius_metrics(result)
    arrays = arrays_from_result(result)
    ts_us = arrays.time_us
    driving_kpa = arrays.pressure_kpa
    radius_um = arrays.radius_um

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.set_ylabel("Driving Pressure (kPa)", color="black")
    ax1.plot(ts_us, driving_kpa, label="Driving Pressure", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.set_xlabel("Time (μs)")
    ax2.set_ylabel("Radius (μm)", color="black")
    ax2.plot(ts_us, radius_um, label="Radius / Time", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    print("Converged:", result.converged)
    for key, value in metrics.items():
        print(f"{key}: {value:.3e}")


if __name__ == "__main__":
    demo()
