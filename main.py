import matplotlib.pyplot as plt
import numpy as np

from jbubble import (
    Units,
    compute_radius_metrics,
    default_bubble,
    default_pulse,
    run_simulation,
)


def demo_run(R0=4e-6, pressure=1e6, freq=800e3):
    units = Units()
    bubble = default_bubble(R0=R0)
    pulse = default_pulse(freq=freq, pressure=pressure)
    result = run_simulation(bubble=bubble, pulse=pulse, units=units)
    metrics = compute_radius_metrics(result)

    ts_us = np.array(result.ts) * 1e6
    driving_kpa = np.array(result.driving_pressure) / 1e3
    radius_um = np.array(result.radius) * 1e6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.set_ylabel("Driving Pressure (kPa)", color="tab:blue")
    ax1.plot(ts_us, driving_kpa, label="Driving Pressure", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.set_xlabel("Time (μs)")
    ax2.set_ylabel("Radius (μm)", color="tab:orange")
    ax2.plot(ts_us, radius_um, label="Radius / Time", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    print("Converged:", result.converged)
    for key, value in metrics.items():
        print(f"{key}: {value:.3e}")


def main():
    demo_run()


if __name__ == "__main__":
    main()
