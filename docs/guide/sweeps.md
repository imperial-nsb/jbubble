# Parameter sweeps

A simulation is a pure JAX function of its parameters, so you can run
thousands of them as one batched program. This page shows how to batch
simulations with `jax.vmap`, how to sweep a grid of parameters in parallel
with [`GridSweep`][jbubble.utils.gridsweep.GridSweep], and how to keep the
results correct, reproducible, and on disk.

Every code block on this page runs as written, in order.

## Batch simulations with `jax.vmap`

Write a function of the parameters that you want to vary, then let
`jax.vmap` map it over arrays of values. The following code simulates a row
of 16 free bubbles from 1 to 5 µm in one call:

```python
import jax
import jax.numpy as jnp

from jbubble import SaveSpec, run_simulation
from jbubble.utils.presets import free_bubble


def peak_ratio(R0):
    eom, pulse = free_bubble(R0=R0, freq=1e6, pressure=100e3)
    result = run_simulation(eom, pulse, save_spec=SaveSpec(256))
    return result.radius.max() / R0, result.converged


radii = jnp.linspace(1e-6, 5e-6, 16)  # [m]
ratios, converged = jax.jit(jax.vmap(peak_ratio))(radii)
print(f"all converged: {bool(converged.all())}")
print(f"largest response: R0 = {radii[ratios.argmax()] * 1e6:.2f} µm")
```

Inside the batch, every simulation takes as many solver steps as the slowest
one, so one bubble that collapses violently, or hits `max_steps`, slows the
whole batch. Return the `converged` flag with each result: under `jax.vmap`,
`run_simulation` can't warn about a failed solve.

## Sweep a grid with `GridSweep`

[`GridSweep`][jbubble.utils.gridsweep.GridSweep] evaluates a function on
every point of a Cartesian grid of named parameters. It compiles
`jax.vmap(fn)` once, splits the grid into chunks, and runs the chunks in
parallel on worker threads, one per CPU core by default. It returns NumPy
arrays shaped like the grid, with one axis per parameter in sorted order of
the names.

The function can return any PyTree. Return the metric with the
`converged` flag and the step count, so you can mask failed points and spot
slow regions of the grid. [`solve_eom`][jbubble.solver.solve_eom] gives
the step count:

```{.python continuation}
import diffrax
import numpy as np

from jbubble import solve_eom
from jbubble.utils.gridsweep import GridSweep


def response(R0, pressure):
    eom, pulse = free_bubble(R0=R0, freq=1e6, pressure=pressure)
    solution = solve_eom(eom, pulse, save_spec=SaveSpec(256))
    return {
        "ratio": solution.ys.R.max() / R0,
        "converged": diffrax.is_successful(solution.result),
        "num_steps": solution.stats["num_steps"],
    }


search_space = {
    "R0": jnp.linspace(1e-6, 6e-6, 24),  # [m]
    "pressure": jnp.linspace(20e3, 300e3, 8),  # [Pa]
}
sweep = GridSweep(response, search_space, progress=False)
grid = sweep.run()

ratio = np.where(grid["converged"], grid["ratio"], np.nan)  # mask failures
print(grid["ratio"].shape)  # (24, 8): R0 by pressure
print(f"{sweep.workers} workers, {sweep.per_worker} points per chunk")
print(f"steps per simulation: {grid['num_steps'].min()} to {grid['num_steps'].max()}")
```

A failed solve returns `inf` samples, so a metric such as the peak radius is
`inf` there. Mask it with `converged` before you plot or average the grid.

For grids too large to hold in memory, iterate over
[`batches`][jbubble.utils.gridsweep.GridSweep.batches] instead of calling
`run`. Each batch is a `(params, outputs)` pair of flat arrays, so you can
stream results to disk or keep a running summary:

```{.python continuation}
best = (0.0, None)
for params, outputs in GridSweep(response, search_space, progress=False).batches():
    ratios = np.where(outputs["converged"], outputs["ratio"], 0.0)
    i = ratios.argmax()
    if ratios[i] > best[0]:
        best = (ratios[i], (params["R0"][i], params["pressure"][i]))
R0_best, p_best = best[1]
print(f"largest R/R0 = {best[0]:.2f} at R0 = {R0_best * 1e6:.2f} µm, {p_best / 1e3:.0f} kPa")
```

If you stop iterating early, for example with `break`, `GridSweep` cancels
the chunks that haven't started.

## Choose workers and devices

| Argument | Default | What it does |
|---|---|---|
| `workers` | One per CPU core available to the process, up to 31 per CPU device; one per device on a GPU or TPU | Number of chunks evaluated at the same time |
| `devices` | Every GPU or TPU; on CPU, as many CPU devices as the workers need, one for up to 31 workers | Devices that run the chunks |
| `batch_size` | `512` | Grid points per batch; each worker evaluates about `batch_size / workers` points per call |

On CPU, worker threads share one JAX device and run in parallel, so you
don't set `XLA_FLAGS` or `JAX_NUM_CPU_DEVICES`. XLA runs at most 32
computations at a time on a CPU device, so to use more than 31 cores, set
`JAX_NUM_CPU_DEVICES` to the number of cores divided by 31, rounded up,
before you import JAX. To leave cores free on a shared machine, or in a
container whose CPU quota Python can't see, pass `workers`:

```{.python continuation}
polite = GridSweep(response, search_space, workers=4, progress=False)
print(polite.workers, polite.per_worker)
```

Worker threads don't inherit JAX settings that a `with` block sets for the
calling thread only, such as `jax.debug_nans(True)`. Set those with
`jax.config.update` instead. Host callbacks inside the function, such as
`jax.debug.print`, can run from several threads at once.

### CPU and GPU

On a CPU, parallelism comes from the worker threads, and each chunk is
small: with 16 workers and the default batch size, 32 points. On a GPU,
each device runs one chunk at a time as a single batched program, so a
chunk of many points keeps the device busy. To give each GPU more work,
raise `batch_size`, for example to a few thousand points. A batched solve
still waits for its slowest member, so stiff or violently collapsing
regions of a grid cost more on a GPU, where a chunk is large, than on a CPU.

A single simulation is a sequential time integration, so it doesn't run
faster on a GPU. To run on the CPU when a GPU is present, set
`JAX_PLATFORMS=cpu` before you import JAX.

## Make sweeps reproducible

For a given chunk size, `per_worker`, the results don't depend on the number
of workers or devices: a parallel sweep is bitwise identical to a serial one
(`workers=1`) with the same chunk size. By default, `per_worker` depends on
the core count and the grid size, and a different chunk size compiles a
different program, which can shift results near a violent collapse by more
than the solver tolerances. To get the same chunks on any machine, pass
`workers` and `batch_size` explicitly:

```{.python continuation}
parallel = GridSweep(response, search_space, batch_size=64, workers=4, progress=False)
serial = GridSweep(response, search_space, batch_size=16, workers=1, progress=False)
assert parallel.per_worker == serial.per_worker == 16
same = np.array_equal(parallel.run()["ratio"], serial.run()["ratio"])
print("bitwise identical:", same)
```

A different CPU, or another version of JAX, can still change the results
slightly.

## Save and load results

[`export_hdf5`][jbubble.utils.io.export_hdf5] writes arrays and a dictionary
of metadata to an HDF5 file, and [`load_hdf5`][jbubble.utils.io.load_hdf5]
reads them back. Both need the `io` extra:

```{.python continuation}
import pathlib
import tempfile

from jbubble.utils.io import export_hdf5, load_hdf5

with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / "sweep.h5"
    export_hdf5(
        path,
        metadata={"preset": "free_bubble", "freq": 1e6},
        R0=np.asarray(search_space["R0"]),
        pressure=np.asarray(search_space["pressure"]),
        **grid,
    )
    arrays, metadata = load_hdf5(path)
print(sorted(arrays), arrays["ratio"].shape, metadata["freq"])
```

For a full sweep with a response map, a linear-resonance overlay, and HDF5
export, see the example
[Parameter sweeps](../examples/05_parameter_sweeps.md).
