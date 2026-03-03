import matplotlib.pyplot as plt

from jbubble import (
    Units,
    SaveSpec,
    arrays_from_result,
    compute_radius_metrics,
    run_simulation,
)
import jbubble.shapes as shapes
from jbubble.bubble import Marmottant
from jbubble.pulse import Pulse
import time
from jax import jit


def demo():
    units = Units()
    bubbles = [
        Marmottant(radius)
        for radius in [
            1e-6,
            2e-6,
            3e-6,
        ]
    ]
    freq = 300e3
    pressure = 100e3
    pulse = Pulse(
        freq=freq,
        pressure=pressure,
        shape=shapes.Sine(),
        cycle_num=4,
        initial_time=1e-6,
        apply_hann=False,
    )

    save_spec = SaveSpec(num_samples=1000)

    # JIT compile the simulation function
    jit_run_simulation = jit(run_simulation)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for idx, bubble in enumerate(bubbles):
        # First run (includes compilation time)
        start = time.perf_counter()
        result = jit_run_simulation(
            bubble=bubble,
            pulse=pulse,
            units=units,
            save_spec=save_spec,
        )
        end = time.perf_counter()

        if idx == 0:
            print(f"First run (+ JIT): {end - start:.4f} seconds")
        else:
            print(f"Subsequent run #{idx + 1}: {end - start:.4f} seconds")

        metrics = compute_radius_metrics(result)
        arrays = arrays_from_result(result)
        radius_um = arrays.radius_um

        ts_us = arrays.time_us
        driving_kpa = arrays.pressure_kpa

        ax2.plot(ts_us, radius_um / bubble.R0, label=f"R / (R0 = {bubble.R0:.1e})")

        print("Converged:", result.converged)
        print("Metrics:")
        for key, value in metrics.items():
            print(f"\t{key}: {value:.3e}")

        print("-" * 40)

    ax1.set_ylabel("Driving Pressure (kPa)", color="black")
    ax1.plot(ts_us, driving_kpa, label="Driving Pressure", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.set_xlabel("Time (μs)")
    ax2.set_ylabel("R / R0", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()
