"""Benchmark GridSweep worker threads against a serial sweep.

The script sweeps the peak expansion ratio of a free bubble (Keller-Miksis,
no shell) over a radius-by-pressure grid, once per worker count, and prints
the first-run time (trace, compile, and run) and the steady-state time of
each configuration.

Every configuration uses the same chunk size (`--chunk` points per call), so
all of them run the same compiled program, and the script checks that their
outputs are bitwise identical. It doesn't assert a speedup: thread scaling
depends on the machine, its core count, and its load.

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

Compare threads on one CPU device with four virtual CPU devices:

```bash
uv run python scripts/bench_gridsweep.py --workers 1 16 --cpu-devices 4
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
        help="expose this many virtual CPU devices and sweep across all of them",
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
    from jbubble.utils.gridsweep import GridSweep, _available_cores
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
    devices = args.cpu_devices
    if args.workers:
        worker_counts = sorted(set(args.workers))
    else:
        worker_counts = sorted(
            {2**k for k in range(int(math.log2(cores)) + 1)} | {cores}
        )
    # More workers than chunks would shrink the chunk, which changes the
    # compiled program: cap the worker count so every row uses --chunk.
    n_chunks = math.ceil(n_r0 * n_p / args.chunk)
    worker_counts = sorted({min(w, n_chunks) for w in worker_counts})
    if devices is not None:
        worker_counts = [w for w in worker_counts if w >= devices]

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
    for workers in worker_counts:
        sweep = GridSweep(
            peak_ratio,
            search_space,
            batch_size=args.chunk * workers,
            progress=False,
            devices=devices,
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
        if reference is None:
            reference = out
        bitwise = repeatable and all(np.array_equal(out[k], reference[k]) for k in out)
        if serial_steady is None:
            serial_steady = steady
        print(
            f"{sweep.workers:>7} {len(sweep.devices):>7} {sweep.per_worker:>5} "
            f"{first:>10.2f} "
            f"{steady:>11.3f} {serial_steady / steady:>7.1f}x "
            f"{sweep.total_points / steady:>9.0f} "
            f"{int(out['ok'].sum()):>5}/{sweep.total_points:<4} "
            f"{'yes' if bitwise else 'NO':>8}"
        )

    print()
    print("speedup: steady-state time of the first row divided by this row's")
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
