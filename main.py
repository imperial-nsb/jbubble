import matplotlib.pyplot as plt
import numpy as np

from jbubble import (
    Units,
    compute_radius_metrics,
    default_bubble,
    default_pulse,
    run_simulation,
)


def demo_run(R0=4e-6, pressure=500e3, freq=800e3):
    units = Units()
    bubbleA = default_bubble(R0=R0)
    bubbleB = default_bubble(R0=R0 * 1.5)
    pulse = default_pulse(freq=freq, pressure=pressure)
    resultA = run_simulation(bubble=bubbleA, pulse=pulse, units=units)
    resultB = run_simulation(bubble=bubbleB, pulse=pulse, units=units)
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
