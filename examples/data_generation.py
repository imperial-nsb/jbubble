"""
Generate CNN training dataset: radiated pressure waveforms from bubble simulations.

Sweeps over (freq, pressure, R0) with fixed shell params. Streams results to disk
batch-by-batch via GridSweep — memory stays O(batch_size) regardless of grid size.

Output
------
  dataset.h5
    datasets : P_rad (N, NUM_SAMPLES) float32
               label (N,) int32
               freq  (N,) float64
               pressure (N,) float64
               R0    (N,) float64
    attrs    : gamma, chi, kappa_s, d_sensor

Performance note
----------------
NUM_SAMPLES is the dominant cost — diffrax records every save point inside the
XLA scan, so 10k pts ≈ 5.8× slower than 1k pts for the same sims.
50 MHz at 20 µs ≡ 1000 pts — likely sufficient for CNNs below ~10 MHz.
"""

from __future__ import annotations

import time

import h5py
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from jbubble import Bubble, Pulse, SaveSpec, Sine, Units, run_simulation
from jbubble.utils import GridSweep

# ── configuration ─────────────────────────────────────────────────────────────

OUTPUT_PATH = "dataset.h5"
BATCH_SIZE  = 2048

# Fixed simulation settings
# ↳ NUM_SAMPLES: dominant perf knob — 10k ≈ 5.8× slower than 1k (diffrax records
#   each save point inside the XLA loop). 1k → 50 MHz @ 20 µs; fine for ≤10 MHz.
NUM_SAMPLES = 1000
UNITS       = Units()
SAVE_SPEC   = SaveSpec(num_samples=NUM_SAMPLES)
WINDOW_S    = 20e-6   # 20 µs

# Fixed shell parameters (Marmottant model)
GAMMA   = 1.07
CHI     = 0.38      # N/m  — shell elasticity
KAPPA_S = 2.4e-9    # kg/s — shell viscosity

# Sensor / material constants — rho_L is taken from the bubble model at runtime
D_SENSOR = 50e-3   # 50 mm

# Inertial-cavitation threshold
IC_THRESHOLD = 2.0

# Parameter sweep axes
SEARCH_SPACE = {
    "freq":     jnp.linspace(1.0e6, 2.0e6, 20),   # 1–2 MHz
    "pressure": jnp.linspace(50e3,  1000e3, 20),   # 50–1000 kPa
    "R0":       jnp.linspace(1e-6,  10e-6,  20),   # 1–10 µm
}

# ── simulation kernel ─────────────────────────────────────────────────────────────

def simulate(freq, pressure, R0):
    """Run one bubble simulation and return waveform + diagnostics."""
    bubble = Bubble(R0=R0, gamma=GAMMA, chi=CHI, kappa_s=KAPPA_S)
    pulse  = Pulse(freq=freq, pressure=pressure, shape=Sine(),
                   cycle_num=4.0, initial_time=1e-6)
    result = run_simulation(bubble=bubble, pulse=pulse,
                            units=UNITS, save_spec=SAVE_SPEC, window_s=WINDOW_S)

    P_rad     = result.radiated_pressure(D_SENSOR)
    max_ratio = jnp.max(result.radius) / R0
    label     = (max_ratio > IC_THRESHOLD).astype(jnp.int32)

    return {
        "P_rad":     P_rad,
        "label":     label,
        "converged": result.converged,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    gs = GridSweep(simulate, SEARCH_SPACE, batch_size=BATCH_SIZE)
    N  = gs.total_points
    print(f"Grid points : {N:,}  |  batch size : {BATCH_SIZE}  |  save pts : {NUM_SAMPLES}")
    print(f"Output      : {OUTPUT_PATH}\n")

    sample_idx = 0
    n_failed   = 0
    t0         = time.time()

    with h5py.File(OUTPUT_PATH, "w") as hf:
        # Fixed params as file-level attributes
        hf.attrs["gamma"]   = GAMMA
        hf.attrs["chi"]     = CHI
        hf.attrs["kappa_s"] = KAPPA_S
        hf.attrs["d_sensor"] = D_SENSOR

        # Pre-allocate datasets sized for the full grid; resize at end if needed
        chunk = (min(64, BATCH_SIZE), NUM_SAMPLES)
        ds_P  = hf.create_dataset("P_rad",    shape=(N, NUM_SAMPLES), maxshape=(None, NUM_SAMPLES), dtype="float32", chunks=chunk)
        ds_lbl = hf.create_dataset("label",   shape=(N,), maxshape=(None,), dtype="int32")
        ds_f   = hf.create_dataset("freq",    shape=(N,), maxshape=(None,), dtype="float64")
        ds_p   = hf.create_dataset("pressure",shape=(N,), maxshape=(None,), dtype="float64")
        ds_r   = hf.create_dataset("R0",      shape=(N,), maxshape=(None,), dtype="float64")

        for params, outputs in gs.batches():
            mask  = np.asarray(outputs["converged"], dtype=bool)
            n_failed += int((~mask).sum())
            if not mask.any():
                continue

            end = sample_idx + mask.sum()
            ds_P[sample_idx:end]   = np.asarray(outputs["P_rad"],    dtype=np.float32)[mask]
            ds_lbl[sample_idx:end] = np.asarray(outputs["label"],    dtype=np.int32)[mask]
            ds_f[sample_idx:end]   = np.asarray(params["freq"],      dtype=np.float64)[mask]
            ds_p[sample_idx:end]   = np.asarray(params["pressure"],  dtype=np.float64)[mask]
            ds_r[sample_idx:end]   = np.asarray(params["R0"],        dtype=np.float64)[mask]
            sample_idx = end

        # Shrink datasets if any sims failed
        if sample_idx < N:
            for ds in [ds_P, ds_lbl, ds_f, ds_p, ds_r]:
                ds.resize(sample_idx, axis=0)

    elapsed = time.time() - t0
    print(f"\nDone — {sample_idx:,} saved, {n_failed} failed, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
