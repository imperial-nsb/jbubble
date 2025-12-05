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


def demo():
    units = Units()
    bubble = default_bubble(R0=3.0e-6)
    freq = 800e3
    pressure = 1e6
    pulse = build_pulse(
        "quadratic",
        freq=freq,
        pressure=pressure,
        cycle_num=10,
        initial_time=1e-6,
        n=3,
        apply_hann=False,
    )

    save_spec = SaveSpec(num_samples=1000)

    result = run_simulation(
        bubble=bubble,
        pulse=pulse,
        units=units,
        save_spec=save_spec,
    )

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
