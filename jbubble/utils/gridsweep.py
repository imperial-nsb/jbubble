"""Batched, parallel sweeps over a Cartesian product of parameter axes.

[`GridSweep`][jbubble.utils.gridsweep.GridSweep] has two public entry
points:

- [`run`][jbubble.utils.gridsweep.GridSweep.run] runs the full sweep and
  returns a grid-shaped PyTree of NumPy arrays with leaves of shape
  `(*grid_shape, *leaf_shape)`.
- [`batches`][jbubble.utils.gridsweep.GridSweep.batches] is a lazy
  iterator that yields `(params, outputs)` one batch at a time, so memory
  stays O(`batch_size`).

You decide what to do with each batch: stream it to disk, accumulate
statistics, track a running argmin, or build a dataset.

Notes
-----
- `GridSweep` traces `fn` once, compiles `jax.vmap(fn)` ahead of time for
  each device it uses, and evaluates fixed-size chunks of the grid
  concurrently from a pool of worker threads. On CPU, up to 31 workers
  share the default CPU device, so you don't set `JAX_NUM_CPU_DEVICES` or
  `XLA_FLAGS` to run in parallel.
- `fn` must be JAX-compatible: `GridSweep` calls it through `jax.jit` and
  `jax.vmap`.
- Results come back to the host as NumPy arrays, in grid order.
- Grid order is row-major (the last axis varies fastest), matching the
  `numpy.unravel_index` conventions.

Examples
--------
```python
import jax.numpy as jnp

from jbubble import run_simulation
from jbubble.utils import GridSweep
from jbubble.utils.presets import free_bubble


def peak_ratio(R0, pressure):
    eom, pulse = free_bubble(R0=R0, pressure=pressure)
    return run_simulation(eom, pulse).radius.max() / R0


radii = jnp.linspace(1e-6, 5e-6, 20)
pressures = jnp.array([50e3, 100e3, 200e3])
gs = GridSweep(peak_ratio, {"R0": radii, "pressure": pressures})

# full sweep: grid-shaped PyTree of NumPy arrays
grid = gs.run()  # shape (20, 3)

# or stream batch by batch
for params, outputs in gs.batches():
    process(params, outputs)
```
"""

from __future__ import annotations

import math
import operator
import os
import queue
import warnings
from collections import deque
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import SingleDeviceSharding
from tqdm import TqdmWarning

# In a Jupyter kernel without ipywidgets, importing tqdm.auto warns that
# IProgress isn't found and falls back to a text bar. The text bar works, so
# don't tell every notebook that imports jbubble.utils to install ipywidgets.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", TqdmWarning)
    from tqdm.auto import tqdm

PyTree = Any

__all__ = ["GridSweep"]

# Prefix of the worker thread names, so you can tell them apart in a debugger
# or a thread dump.
_THREAD_NAME_PREFIX = "jbubble-gridsweep"

# XLA's CPU client runs at most 32 computations at a time on each CPU device
# (`max_inflight_computations_per_device = 32` in openxla/xla
# xla/pjrt/plugin/xla_cpu/cpu_client_options.h), and JAX has no option to
# change it. Further computations wait for a free slot. A host callback inside
# `fn` that runs a JAX operation needs a slot of its own, so if all 32 slots
# belonged to chunks waiting in such a callback, the sweep would deadlock.
# Running at most 31 chunks per CPU device leaves one slot free; more than 32
# would add no parallelism anyway.
_MAX_CHUNKS_PER_CPU_DEVICE = 31


def _available_cores() -> int:
    """Return the number of CPU cores that this process may run on."""
    # Python 3.13+: honours the CPU affinity mask and PYTHON_CPU_COUNT.
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        n = process_cpu_count()
        if n:
            return int(n)
    # Python 3.12 on Linux: the CPU affinity mask.
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            n = len(sched_getaffinity(0))
        except OSError:
            n = 0
        if n:
            return n
    return os.cpu_count() or 1


def _resolve_devices(devices: int | Sequence[jax.Device] | None) -> list[jax.Device]:
    """Turn the `devices` argument into a list of distinct JAX devices.

    `None` gives every local device. On CPU, `_resolve_devices_and_workers`
    then keeps only as many as the workers need.
    """
    local = jax.local_devices()
    if devices is None:
        return list(local)
    if isinstance(devices, bool):
        raise TypeError("devices must be None, an int, or a sequence of devices.")
    if isinstance(devices, int | np.integer):
        n = int(devices)
        if not 1 <= n <= len(local):
            raise ValueError(
                f"devices={n}, but {len(local)} local {local[0].platform} "
                "device(s) are available. On CPU, set `workers` instead: worker "
                "threads run in parallel on a single CPU device."
            )
        return list(local[:n])
    resolved = list(devices)
    if not resolved:
        raise ValueError("devices must not be an empty sequence.")
    if not all(isinstance(d, jax.Device) for d in resolved):
        raise TypeError("devices must contain only jax.Device objects.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("devices must not contain the same device twice.")
    return resolved


def _resolve_devices_and_workers(
    devices: int | Sequence[jax.Device] | None, workers: int | None
) -> tuple[list[jax.Device], int]:
    """Turn the `devices` and `workers` arguments into a device list and a count.

    On CPU, at most `_MAX_CHUNKS_PER_CPU_DEVICE` workers share each device.
    An explicit `workers` above that limit is lowered to it, with a warning.
    """
    local = jax.local_devices()
    # Extra virtual CPU devices share the same cores, so on CPU `devices=None`
    # uses only as many devices as the workers need, one per
    # _MAX_CHUNKS_PER_CPU_DEVICE workers, which is one device on most
    # machines. Each device costs a compile. Each accelerator brings its own
    # hardware, so `devices=None` uses all of them.
    auto_cpu = devices is None and local[0].platform == "cpu"
    resolved = _resolve_devices(devices)
    limit = (
        _MAX_CHUNKS_PER_CPU_DEVICE * len(resolved)
        if any(d.platform == "cpu" for d in resolved)
        else None
    )

    if workers is None:
        if all(d.platform == "cpu" for d in resolved):
            workers = min(
                _available_cores(), _MAX_CHUNKS_PER_CPU_DEVICE * len(resolved)
            )
            if not auto_cpu:
                workers = max(workers, len(resolved))
        else:
            workers = len(resolved)
    else:
        workers = operator.index(workers)
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}.")
        if limit is not None and workers > limit:
            needed = math.ceil(workers / _MAX_CHUNKS_PER_CPU_DEVICE)
            if auto_cpu:
                how = f"set `JAX_NUM_CPU_DEVICES={needed}` before you import JAX"
            elif len(jax.local_devices(backend="cpu")) >= needed:
                how = f"pass `devices={needed}`"
            else:
                how = (
                    f"set `JAX_NUM_CPU_DEVICES={needed}` before you import JAX "
                    f"and pass `devices={needed}`"
                )
            warnings.warn(
                f"workers={workers}, but {len(resolved)} CPU device(s) can run "
                f"at most {limit} chunks at a time without risking a deadlock, "
                f"so GridSweep uses {limit} workers. XLA runs at most 32 "
                "computations at a time on each CPU device. To run "
                f"{workers} workers, {how}.",
                stacklevel=3,
            )
            workers = limit

    if auto_cpu:
        resolved = resolved[: math.ceil(workers / _MAX_CHUNKS_PER_CPU_DEVICE)]
    elif workers < len(resolved):
        raise ValueError(
            f"workers={workers}, but the sweep uses {len(resolved)} "
            "devices: each device needs at least one worker."
        )
    return resolved, workers


class GridSweep:
    """Batched, parallel sweep over a Cartesian product of named parameter axes.

    `GridSweep` splits the grid into chunks of `per_worker` points and
    evaluates `jax.vmap(fn)` on each chunk. It traces `fn` once, compiles one
    executable per device ahead of time, and runs the chunks concurrently
    from `workers` threads. Results come back to the host as NumPy arrays,
    in grid order.

    Parameters
    ----------
    fn : callable
        `fn(**params) -> PyTree`. `GridSweep` calls it with scalar keyword
        arguments, one per axis name, so it must be JAX-vmappable and
        jit-able.
    search_space : dict[str, jax.Array]
        `{param_name: 1-D array of values}`. The sweep covers the
        Cartesian product of all axes, in `sorted()` order of the names
        (Unicode code-point order, so uppercase names sort before lowercase
        ones).
    batch_size : int
        Target number of grid points in each batch that
        [`batches`][jbubble.utils.gridsweep.GridSweep.batches] yields. A
        batch holds one chunk per worker, and each worker evaluates
        `ceil(min(batch_size, total_points) / workers)` points per call.
        The batch size that `GridSweep` uses, in the `batch_size`
        attribute, is therefore at least `min(batch_size, total_points)`
        and less than that plus `workers`, and a grid with at most
        `batch_size` points comes in one batch. Default: `512`.
    progress : bool
        Whether to show a `tqdm` progress bar during iteration.
        Default: `True`.
    devices : int or sequence of jax.Device, optional
        Devices to run on. `None` uses every local device on GPU or TPU. On
        CPU, it uses one local CPU device per 31 workers: the default CPU
        device for up to 31 workers, and more only if you set
        `JAX_NUM_CPU_DEVICES`. An `int` `n` uses the first `n` local
        devices, and a sequence uses exactly those devices.
        Default: `None`.
    workers : int, optional
        Number of chunks evaluated at the same time, each from its own
        thread. `None` uses one worker per CPU core available to the
        process (respecting its CPU affinity), up to 31 per CPU device, on
        CPU, and one worker per device on GPU or TPU. `1` evaluates the
        chunks one after another on the calling thread. `workers` must be
        at least the number of devices. On CPU, `GridSweep` lowers a value
        above 31 per device to that limit; see the parallelism note. The
        default follows the core count, so it also sets `per_worker`; see
        the reproducibility note. Default: `None`.

    Attributes
    ----------
    devices : list[jax.Device]
        Devices that the sweep runs on. A grid with fewer chunks than
        devices uses only as many devices as it has chunks.
    workers : int
        Number of chunks evaluated at the same time. It's at most the
        number of chunks in the grid.
    per_worker : int
        Number of grid points in each chunk, that is, in each call to the
        compiled `jax.vmap(fn)`.
    batch_size : int
        Number of grid points in each batch: `per_worker * workers`.

    Raises
    ------
    ValueError
        If `search_space` is empty or one of its axes isn't a non-empty
        1-D array, if `batch_size` or `workers` is less than 1, if
        `devices` asks for more devices than are available or repeats a
        device, or if `workers` is less than the number of devices.
    TypeError
        If `fn` isn't callable, a name in `search_space` isn't a string,
        `batch_size` or `workers` isn't an integer, or `devices` has the
        wrong type.

    Warns
    -----
    UserWarning
        If `workers` is more than 31 per CPU device. `GridSweep` then runs
        31 workers per CPU device.

    Notes
    -----
    - **Parallelism.** On CPU, one JAX device runs computations from
      several threads in parallel, so up to 31 workers need no
      `JAX_NUM_CPU_DEVICES` or `XLA_FLAGS` setup. XLA runs at most 32
      computations at a time on each CPU device, and a host callback that
      runs a JAX operation needs a free slot, so `GridSweep` runs at most
      31 chunks at a time on each CPU device. To use more than 31 cores,
      set `JAX_NUM_CPU_DEVICES` to `ceil(cores / 31)` before you import
      JAX; `devices=None` then uses as many CPU devices as the workers
      need. Below 32 workers, extra virtual CPU devices only add compiles.
      To leave cores free on a shared machine, or in a container whose CPU
      quota Python can't see, set `workers`.
    - **Compilation.** Every call has the same shape, `per_worker` points,
      so `fn` is traced once and compiled once per device, and a ragged
      last chunk doesn't trigger a recompile. `GridSweep` pads that chunk
      with copies of its last grid point, never with synthetic values
      such as zeros, and drops the padding from the output.
    - **Load balancing.** Up to `2 * workers` chunks are in flight at a
      time, across batch boundaries, so a slow chunk holds up only its own
      worker. Within a chunk, `jax.vmap` runs the solver loop until its
      slowest point finishes, so a point that hits `max_steps` still
      slows its whole chunk. Check `converged` in the results.
    - **Reproducibility.** For a given `per_worker`, results don't depend
      on `workers` or `devices`: a sweep is bitwise identical to a serial
      sweep (`workers=1`) with the same `per_worker`. `per_worker` is
      `ceil(min(batch_size, total_points) / workers)`, so by default it
      depends on the core count and the grid size. A different chunk size
      compiles a different program, which can shift saved trajectories
      near violent collapses by more than the solver tolerances. To get
      the same chunks on another machine, pass `workers` and `batch_size`
      explicitly. A different CPU or JAX version can still change the
      results slightly.
    - **Stopping early.** If you stop iterating over
      [`batches`][jbubble.utils.gridsweep.GridSweep.batches] early, for
      example with `break`, an exception, or Ctrl+C, `GridSweep` cancels
      the chunks that haven't started and waits for the running ones to
      finish.
    - **Memory.** Device memory holds at most `workers` chunks at a time,
      and host memory holds at most `2 * workers` finished chunks plus the
      current batch.
    - **Thread safety.** Host callbacks inside `fn`, such as
      `jax.debug.callback` or `jax.debug.print`, can run at the same time
      from several threads, so they must be thread-safe. A host callback
      must not run JAX operations, such as comparing or reducing the
      `jax.Array` arguments of a `jax.debug.callback`; convert them with
      `np.asarray` first. Such an operation needs a free computation slot
      on the device, so it's slow, and it can deadlock if other JAX
      computations in the process fill the slots. Worker threads don't
      inherit JAX settings that a `with` block sets for the calling thread
      only, such as `jax.enable_x64(...)` or `jax.debug_nans(...)`, so
      with more than one worker, the chunks run without them. Set them
      with `jax.config.update` instead.
    - **Known issue.** From JAX 0.11.1, XLA on CPU can deadlock when at
      least as many large FFTs inside a loop run at the same time as XLA
      has threads in its pool
      ([jax-ml/jax#41265](https://github.com/jax-ml/jax/issues/41265)),
      and chunks that run at the same time add to that count. If `fn` runs
      FFTs inside the ODE solve and the sweep stops making progress, set
      `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` before you import
      JAX, as the issue suggests; each FFT then runs on one thread. To
      keep FFTs multithreaded, use the issue's other workaround: before
      you import JAX, set `PJRT_NPROC` to more than the number of FFTs in
      flight, which gives the pool more threads. Fewer `workers` also
      means fewer FFTs in flight, but that doesn't help when a single
      chunk runs enough FFTs at the same time.

    Examples
    --------
    Sweep the peak expansion ratio of a free bubble over radius and
    pressure on every available core:

    ```python
    import jax.numpy as jnp

    from jbubble import run_simulation
    from jbubble.utils.gridsweep import GridSweep
    from jbubble.utils.presets import free_bubble


    def peak_ratio(R0, pressure):
        eom, pulse = free_bubble(R0=R0, pressure=pressure)
        result = run_simulation(eom, pulse)
        return {"ratio": result.radius.max() / R0, "ok": result.converged}


    search_space = {
        "R0": jnp.linspace(1e-6, 5e-6, 20),
        "pressure": jnp.array([50e3, 200e3]),
    }
    grid = GridSweep(peak_ratio, search_space).run()
    print(grid["ratio"].shape)  # (20, 2), a NumPy array
    print(grid["ok"].all())  # True if every simulation converged
    ```

    To leave cores free on a shared machine, limit the worker count:

    ```python
    grid = GridSweep(peak_ratio, search_space, workers=4).run()
    ```
    """

    def __init__(
        self,
        fn: Callable[..., PyTree],
        search_space: Mapping[str, Any],
        batch_size: int = 512,
        progress: bool = True,
        devices: int | Sequence[jax.Device] | None = None,
        workers: int | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}.")
        if not search_space:
            raise ValueError("search_space must have at least one axis.")
        if not all(isinstance(k, str) for k in search_space):
            raise TypeError("search_space names must be strings.")
        batch_size = operator.index(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")

        self.fn = fn
        self.search_space = search_space
        self.progress = progress

        # Stable ordering so multi-index resolution is deterministic
        self._keys = sorted(search_space)
        self._axes = [jnp.asarray(search_space[k]) for k in self._keys]
        for k, a in zip(self._keys, self._axes, strict=True):
            if a.ndim != 1 or a.shape[0] == 0:
                raise ValueError(
                    f"search_space[{k!r}] must be a non-empty 1-D array, "
                    f"got shape {a.shape}."
                )
        # Host copies: GridSweep gathers each chunk's parameters with NumPy.
        self._host_axes = [np.asarray(a) for a in self._axes]
        self._sizes = [a.shape[0] for a in self._host_axes]
        self._N = math.prod(self._sizes)

        resolved, workers = _resolve_devices_and_workers(devices, workers)
        self.per_worker = math.ceil(min(batch_size, self._N) / workers)
        self.workers = min(workers, math.ceil(self._N / self.per_worker))
        self.batch_size = self.per_worker * self.workers
        # Don't compile for devices that a small grid leaves idle.
        self.devices = resolved[: self.workers]

        self._vmapped = jax.jit(jax.vmap(self._call))
        self._executables: dict[jax.Device, Any] = {}

    def _call(self, params: dict[str, jax.Array]) -> PyTree:
        return self.fn(**params)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the full parameter grid, one int per axis."""
        return tuple(self._sizes)

    @property
    def total_points(self) -> int:
        """Total number of grid points."""
        return self._N

    @property
    def num_batches(self) -> int:
        """Number of batches that `batches` yields."""
        return math.ceil(self._N / self.batch_size)

    @property
    def axes(self) -> dict[str, jax.Array]:
        """Parameter axes in sweep order, that is, `sorted()` order of the names."""
        return dict(zip(self._keys, self._axes, strict=True))

    # ── internals ───────────────────────────────────────────────────────────

    def _gather(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """Return the parameter values at the given flat grid indices."""
        multi = np.unravel_index(flat, self._sizes)
        return {k: self._host_axes[i][multi[i]] for i, k in enumerate(self._keys)}

    def _compile(self) -> None:
        """Compile the vmapped `fn` for each device, one device at a time.

        Every lowering has the same abstract input shapes, so JAX traces
        `fn` only once. Compiling sequentially bounds the peak memory.
        """
        for device in self.devices:
            if device in self._executables:
                continue
            sharding = SingleDeviceSharding(device)
            spec = {
                k: jax.ShapeDtypeStruct((self.per_worker,), a.dtype, sharding=sharding)
                for k, a in zip(self._keys, self._host_axes, strict=True)
            }
            self._executables[device] = self._vmapped.lower(spec).compile()

    def _chunk_size(self, c: int) -> int:
        """Return the number of real (unpadded) grid points in chunk `c`."""
        return min(self.per_worker, self._N - c * self.per_worker)

    def _run_chunk(self, c: int, device: jax.Device) -> PyTree:
        """Evaluate chunk `c` on `device` and return its outputs on the host."""
        start = c * self.per_worker
        n = self._chunk_size(c)
        # Pad a ragged chunk with copies of its last real point: a synthetic
        # value, such as R0 = 0, could hit max_steps and stall the chunk.
        flat = np.minimum(np.arange(start, start + self.per_worker), start + n - 1)
        args = jax.device_put(self._gather(flat), device)
        out = jax.device_get(self._executables[device](args))
        return jax.tree.map(lambda x: x[:n], out)

    def _chunks(self) -> Generator[PyTree, None, None]:
        """Yield the chunk outputs in grid order.

        With more than one worker, a thread pool keeps up to `2 * workers`
        chunks in flight, across batch boundaries. Closing the generator
        cancels the chunks that haven't started and waits for the running
        ones.
        """
        n_chunks = math.ceil(self._N / self.per_worker)
        if self.workers == 1:
            for c in range(n_chunks):
                yield self._run_chunk(c, self.devices[0])
            return

        # One token per worker, spread round-robin over the devices. A chunk
        # takes a token for as long as it runs, so no device runs more than
        # its share of the workers at a time.
        tokens: queue.SimpleQueue[jax.Device] = queue.SimpleQueue()
        for i in range(self.workers):
            tokens.put(self.devices[i % len(self.devices)])

        def task(c: int) -> PyTree:
            device = tokens.get()
            try:
                return self._run_chunk(c, device)
            finally:
                tokens.put(device)

        pool = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix=_THREAD_NAME_PREFIX
        )
        window: deque[Future[PyTree]] = deque()
        try:
            submitted = 0
            while submitted < n_chunks or window:
                while submitted < n_chunks and len(window) < 2 * self.workers:
                    window.append(pool.submit(task, submitted))
                    submitted += 1
                yield window.popleft().result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _progress_chunks(self) -> Generator[PyTree, None, None]:
        """Compile, then yield the chunk outputs while updating a progress bar."""
        compiled = all(d in self._executables for d in self.devices)
        desc = "Grid sweep" if compiled else "Grid sweep (compiling)"
        with tqdm(
            total=self._N, desc=desc, unit="pt", disable=not self.progress
        ) as bar:
            if not compiled:
                self._compile()
                bar.set_description_str("Grid sweep")
            chunks = self._chunks()
            try:
                for c, out in enumerate(chunks):
                    bar.update(self._chunk_size(c))
                    yield out
            finally:
                # Close the inner generator explicitly so the thread pool
                # stops even where garbage collection is lazy.
                chunks.close()

    # ── public API ──────────────────────────────────────────────────────────

    def batches(
        self,
    ) -> Generator[tuple[dict[str, np.ndarray], PyTree], None, None]:
        """Iterate lazily over the grid, one batch at a time.

        The first iteration compiles `fn`. Batches arrive in grid order;
        each has `batch_size` points except possibly the last.

        Yields
        ------
        params : dict[str, numpy.ndarray]
            Batch of parameter values: one array per axis name, with one
            value per grid point in the batch.
        outputs : PyTree
            Corresponding outputs of `fn`, as NumPy arrays with a leading
            batch axis.
        """
        buffer: list[PyTree] = []
        start = 0
        chunks = self._progress_chunks()
        try:
            for out in chunks:
                buffer.append(out)
                end = min(start + self.per_worker * len(buffer), self._N)
                if len(buffer) == self.workers or end == self._N:
                    params = self._gather(np.arange(start, end))
                    outputs = jax.tree.map(lambda *xs: np.concatenate(xs), *buffer)
                    buffer, start = [], end
                    yield params, outputs
        finally:
            chunks.close()

    def run(self) -> PyTree:
        """Run the full sweep and return a grid-shaped PyTree.

        Returns
        -------
        PyTree
            The structure that `fn` returns, with NumPy-array leaves of
            shape `(*grid_shape, *leaf_shape)`. `grid_shape` follows the axes
            in `search_space` in `sorted()` order of the names (Unicode
            code-point order, so uppercase names sort before lowercase
            ones). The grid is row-major: the last axis varies fastest.
        """
        # Fill preallocated arrays chunk by chunk, so the host never holds the
        # result twice.
        grid: list[np.ndarray] = []
        treedef = None
        start = 0
        chunks = self._progress_chunks()
        try:
            for c, out in enumerate(chunks):
                leaves, treedef = jax.tree.flatten(out)
                if c == 0:
                    grid = [np.empty((self._N, *x.shape[1:]), x.dtype) for x in leaves]
                end = start + self._chunk_size(c)
                for dst, src in zip(grid, leaves, strict=True):
                    dst[start:end] = src
                start = end
        finally:
            chunks.close()
        shape = self.grid_shape
        return jax.tree.unflatten(
            treedef, [x.reshape(*shape, *x.shape[1:]) for x in grid]
        )
