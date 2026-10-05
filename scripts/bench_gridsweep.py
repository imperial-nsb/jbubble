"""Benchmark GridSweep worker threads against a serial sweep.

The script sweeps the peak expansion ratio of a free bubble (Keller-Miksis,
no shell) over a radius-by-pressure grid, once per worker count, and prints
the first-run time (trace, compile, and run) and the steady-state time of
each configuration.

Every configuration uses the same chunk size (`--chunk` points per call), so
all of them run the same compiled program, and the script checks that their
outputs are bitwise identical. To keep the chunk size, the script lowers a
worker count that would shrink it: more workers than `points // chunk`, or
more than 31 per CPU device, where `GridSweep` stops. It prints a note for
every count that it lowers or skips, and marks a row whose chunk size still
differs as `n/a` in the bitwise column. It doesn't assert a speedup: thread
scaling depends on the machine, its core count, and its load.

Examples
--------
Compare the default worker counts with Dopri5:

```bash
uv run python scripts/bench_gridsweep.py
```

Benchmark the implicit Kvaerno5 solver on a smaller grid:

```bash
uv run python scripts/bench_gridsweep.py --solver kvaerno5 --points 256
```

Compare threads on one CPU device with the same threads spread over four
virtual CPU devices. Each worker count runs once with `devices=1` and once
with `devices=4`:

```bash
uv run python scripts/bench_gridsweep.py --workers 4 16 --cpu-devices 4
```
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GridSweep worker threads against a serial sweep."
    )
    parser.add_argument(
        "--points", type=int, default=1024, help="grid points (default: 1024)"
    )
    parser.add_argument(
        "--solver",
        choices=["dopri5", "kvaerno5"],
        default="dopri5",
        help="ODE solver (default: dopri5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        help="worker counts to compare (default: 1, 2, 4, ... up to the cores)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=32,
        help="grid points per worker call (default: 32)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="steady-state repeats; the minimum is reported (default: 3)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="saved time samples per simulation (default: 64)",
    )
    parser.add_argument(
        "--cpu-devices",
        type=int,
        default=None,
        help=(
            "expose this many virtual CPU devices, and run each worker count "
            "on one device and on all of them"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cpu_devices is not None:
        # Must happen before JAX starts its CPU backend.
        os.environ["JAX_NUM_CPU_DEVICES"] = str(args.cpu_devices)

    import diffrax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jbubble import SaveSpec, SolverConfig, run_simulation
    from jbubble.utils.gridsweep import (
        _MAX_CHUNKS_PER_CPU_DEVICE,
        GridSweep,
        _available_cores,
    )
    from jbubble.utils.presets import free_bubble

    solver = {"dopri5": diffrax.Dopri5, "kvaerno5": diffrax.Kvaerno5}[args.solver]
    config = SolverConfig(solver=solver())
    save_spec = SaveSpec(args.samples)

    def peak_ratio(R0, pressure):
        eom, pulse = free_bubble(R0=R0, pressure=pressure)
        result = run_simulation(eom, pulse, save_spec=save_spec, config=config)
        return {"ratio": jnp.max(result.radius) / R0, "ok": result.converged}

    n_r0 = max(1, round(math.sqrt(args.points)))
    n_p = max(1, args.points // n_r0)
    search_space = {
        "R0": jnp.linspace(1.5e-6, 4e-6, n_r0),
        "pressure": jnp.linspace(20e3, 300e3, n_p),
    }

    cores = _available_cores()
    if args.workers:
        worker_counts = sorted(set(args.workers))
    else:
        worker_counts = sorted(
            {2**k for k in range(int(math.log2(cores)) + 1)} | {cores}
        )
    device_counts = sorted({1, args.cpu_devices or 1})

    # A row runs `workers` chunks of exactly --chunk points only if the grid
    # has at least that many full chunks, and if GridSweep doesn't lower the
    # worker count (it runs at most 31 chunks per CPU device). Lower any
    # worker count that would shrink the chunk, which changes the program.
    full_chunks = max(1, (n_r0 * n_p) // args.chunk)
    rows: list[tuple[int, int]] = []
    notes: list[str] = []
    for n_devices in device_counts:
        limit = min(full_chunks, _MAX_CHUNKS_PER_CPU_DEVICE * n_devices)
        for requested in worker_counts:
            if requested < n_devices:
                notes.append(
                    f"skipped {requested} worker(s) on {n_devices} devices: "
                    "each device needs at least one worker"
                )
                continue
            workers = min(requested, limit)
            if workers != requested:
                reason = (
                    f"to keep {args.chunk} points per call"
                    if limit == full_chunks
                    else "because GridSweep runs at most "
                    f"{_MAX_CHUNKS_PER_CPU_DEVICE} per CPU device"
                )
                notes.append(
                    f"lowered {requested} workers to {workers} on "
                    f"{n_devices} device(s) {reason}"
                )
            if workers < n_devices:
                notes.append(
                    f"skipped {n_devices} devices: the grid has only "
                    f"{full_chunks} chunk(s) of {args.chunk} points"
                )
            elif (workers, n_devices) not in rows:
                rows.append((workers, n_devices))

    print(f"Python {platform.python_version()}, JAX {jax.__version__}, {sys.platform}")
    print(f"devices: {jax.local_devices()}")
    print(f"available cores: {cores}")
    print(
        f"grid: {n_r0} x {n_p} = {n_r0 * n_p} points, solver {args.solver}, "
        f"{args.chunk} points per call, {args.samples} samples"
    )
    print()
    header = (
        f"{'workers':>7} {'devices':>7} {'chunk':>5} {'first [s]':>10} {'steady [s]':>11} "
        f"{'speedup':>8} {'points/s':>9} {'converged':>10} {'bitwise':>8}"
    )
    print(header)
    print("-" * len(header))

    reference = None
    serial_steady = None
    for workers, n_devices in rows:
        sweep = GridSweep(
            peak_ratio,
            search_space,
            batch_size=args.chunk * workers,
            progress=False,
            devices=n_devices,
            workers=workers,
        )
        start = time.perf_counter()
        out = sweep.run()
        first = time.perf_counter() - start
        steady = math.inf
        for _ in range(args.repeats):
            start = time.perf_counter()
            again = sweep.run()
            steady = min(steady, time.perf_counter() - start)
        repeatable = all(np.array_equal(out[k], again[k]) for k in out)
        if sweep.per_worker != args.chunk:
            # A different chunk size compiles a different program, so the
            # bits may differ: don't compare this row.
            bitwise = "n/a"
        else:
            if reference is None:
                reference = out
            same = all(np.array_equal(out[k], reference[k]) for k in out)
            bitwise = "yes" if repeatable and same else "NO"
        if serial_steady is None:
            serial_steady = steady
        print(
            f"{sweep.workers:>7} {len(sweep.devices):>7} {sweep.per_worker:>5} "
            f"{first:>10.2f} "
            f"{steady:>11.3f} {serial_steady / steady:>7.1f}x "
            f"{sweep.total_points / steady:>9.0f} "
            f"{int(out['ok'].sum()):>5}/{sweep.total_points:<4} "
            f"{bitwise:>8}"
        )

    print()
    print("speedup: steady-state time of the first row divided by this row's")
    for note in notes:
        print(f"note: {note}")
    try:
        import resource
    except ImportError:  # Windows
        return
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss / 2**30 if sys.platform == "darwin" else rss / 2**20
    print(f"peak RSS of the whole run: {rss_gb:.2f} GB")


if __name__ == "__main__":
    main()
