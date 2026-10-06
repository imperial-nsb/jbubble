"""Tests for jbubble.utils.gridsweep.

Tests that need several JAX devices, or that could deadlock XLA, run in a
child process that sets `JAX_NUM_CPU_DEVICES`. The child asserts the device
count, so these tests never skip silently, and the parent kills a child that
hangs. Every test also carries a pytest-timeout limit with the `thread`
method, which dumps the stacks and stops the run when a test hangs. The
default `signal` method can't stop a deadlocked worker thread: its exception
reaches the main thread, which then waits for that worker forever when it
shuts down the thread pool.
"""

import collections
import json
import math
import os
import subprocess
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jbubble import SaveSpec, SolverConfig, run_simulation
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.utils.gridsweep import GridSweep

pytestmark = pytest.mark.timeout(300, method="thread")

SS = {"x": jnp.arange(5.0), "y": jnp.arange(3.0)}  # 15 points: not a power of 2
THREAD_PREFIX = "jbubble-gridsweep"
MAX_PER_CPU_DEVICE = 31  # XLA runs at most 32 computations per CPU device

# Tests of the CPU defaults: on a GPU or TPU host, `workers=None` and
# `devices=None` follow the accelerators instead.
cpu_only = pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="tests the defaults on CPU"
)


def _toy(x, y):
    return {"s": x + y, "p": x * y}


def _assert_toy_grid(out):
    xx, yy = np.meshgrid(np.arange(5.0), np.arange(3.0), indexing="ij")
    np.testing.assert_array_equal(out["s"], xx + yy)
    np.testing.assert_array_equal(out["p"], xx * yy)


def _sweep_threads_alive() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(THREAD_PREFIX)]


def _keller_miksis(R0, pressure):
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=1e-3),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )
    pulse = ToneBurst(freq=1e6, pressure=pressure, shape=Sine(), cycle_num=5)
    return eom, pulse


def _decay_kvaerno5(k):
    """Solve dy/dt = -k y with Kvaerno5, which runs lineax linear solves."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: -args * y),
        diffrax.Kvaerno5(),
        t0=0.0,
        t1=1.0,
        dt0=1e-3,
        y0=jnp.asarray(1.0),
        args=k,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        saveat=diffrax.SaveAt(t1=True),
    )
    return sol.ys[-1]


# ── grid geometry ───────────────────────────────────────────────────────────


class TestGridGeometry:
    def test_grid_shape(self):
        gs = GridSweep(_toy, {"x": jnp.arange(3.0), "y": jnp.arange(2.0)})
        assert gs.grid_shape == (3, 2)

    def test_total_points(self):
        gs = GridSweep(_toy, {"x": jnp.arange(3.0), "y": jnp.arange(2.0)})
        assert gs.total_points == 6

    def test_axes_are_sorted_by_name(self):
        ss = {"b": jnp.arange(2.0), "a": jnp.arange(3.0)}
        gs = GridSweep(lambda b, a: a - b, ss, progress=False)
        assert list(gs.axes) == ["a", "b"]
        assert gs.grid_shape == (3, 2)
        out = gs.run()
        np.testing.assert_array_equal(out, [[0, -1], [1, 0], [2, 1]])

    @pytest.mark.parametrize(
        ("n", "batch_size", "workers", "per_worker", "eff_workers", "eff_batch"),
        [
            (15, 5, 2, 3, 2, 6),  # rounded up to a multiple of workers
            (15, 4, 1, 4, 1, 4),  # serial: batch_size unchanged
            (15, 512, 4, 4, 4, 16),  # grid smaller than batch_size
            (3, 512, 16, 1, 3, 3),  # fewer points than workers
            (1000, 512, 16, 32, 16, 512),
        ],
    )
    def test_batch_size_rounds_up_to_a_multiple_of_workers(
        self, n, batch_size, workers, per_worker, eff_workers, eff_batch
    ):
        gs = GridSweep(
            _toy,
            {"x": jnp.arange(float(n)), "y": jnp.zeros(1)},
            batch_size=batch_size,
            workers=workers,
        )
        assert (gs.per_worker, gs.workers, gs.batch_size) == (
            per_worker,
            eff_workers,
            eff_batch,
        )
        assert gs.num_batches == -(-n // eff_batch)

    def test_batch_size_attribute_stays_within_the_documented_bounds(self):
        for n in (1, 2, 5, 15, 37):
            ss = {"x": jnp.arange(float(n))}
            for batch_size in (1, 3, 5, 8, 512):
                for workers in (1, 2, 3, 4, 7):
                    gs = GridSweep(_toy_1d, ss, batch_size=batch_size, workers=workers)
                    target = min(batch_size, n)
                    assert gs.per_worker == math.ceil(target / workers)
                    assert target <= gs.batch_size < target + workers
                    if n <= batch_size:
                        assert gs.num_batches == 1


# ── values, order, and output types ─────────────────────────────────────────


@pytest.mark.parametrize("workers", [1, 2, 3, 16])
@pytest.mark.parametrize("batch_size", [1, 4, 7, 512])
def test_run_returns_grid_in_row_major_order(workers, batch_size):
    gs = GridSweep(_toy, SS, batch_size=batch_size, progress=False, workers=workers)
    _assert_toy_grid(gs.run())


def test_run_returns_host_numpy_arrays():
    out = GridSweep(_toy, SS, progress=False, workers=2).run()
    for leaf in jax.tree.leaves(out):
        assert type(leaf) is np.ndarray
        assert leaf.flags.writeable


def test_vector_output_keeps_trailing_axes():
    gs = GridSweep(
        lambda x: jnp.array([x, x**2]),
        {"x": jnp.arange(1.0, 4.0)},
        progress=False,
        workers=2,
    )
    out = gs.run()
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out[1], [2.0, 4.0])


def test_pytree_output_with_none_leaf():
    gs = GridSweep(
        lambda x: {"a": x, "b": None, "c": (x, 2 * x)},
        {"x": jnp.arange(4.0)},
        progress=False,
        workers=3,
    )
    out = gs.run()
    assert out["b"] is None
    np.testing.assert_array_equal(out["c"][1], [0.0, 2.0, 4.0, 6.0])


def test_integer_axis_keeps_its_dtype():
    out = GridSweep(lambda n: n * 2, {"n": jnp.arange(5)}, progress=False).run()
    assert np.issubdtype(out.dtype, np.integer)
    np.testing.assert_array_equal(out, [0, 2, 4, 6, 8])


@pytest.mark.parametrize("workers", [1, 4])
def test_batches_cover_grid_once_in_order(workers):
    gs = GridSweep(_toy, SS, batch_size=4, progress=False, workers=workers)
    seen = []
    n_batches = 0
    for params, out in gs.batches():
        for v in (*params.values(), *out.values()):
            assert type(v) is np.ndarray
        np.testing.assert_array_equal(out["s"], params["x"] + params["y"])
        assert len(params["x"]) == min(gs.batch_size, 15 - len(seen))
        seen += list(zip(params["x"].tolist(), params["y"].tolist(), strict=True))
        n_batches += 1
    expected = [(float(x), float(y)) for x in range(5) for y in range(3)]
    assert seen == expected
    assert n_batches == gs.num_batches


# ── compilation and padding ─────────────────────────────────────────────────


@pytest.mark.parametrize("workers", [1, 4])
def test_fn_is_traced_once_despite_ragged_last_chunk(workers):
    traces = []

    def fn(x):
        traces.append(1)  # runs at trace time only
        return x**2

    gs = GridSweep(
        fn, {"x": jnp.arange(10.0)}, batch_size=4, progress=False, workers=workers
    )
    gs.run()
    gs.run()
    list(gs.batches())
    assert len(traces) == 1


def test_ragged_last_chunk_is_padded_with_its_last_point():
    evaluated = []

    def record(x):
        evaluated.extend(np.atleast_1d(x).ravel().tolist())

    def fn(x):
        jax.debug.callback(record, x)
        return x

    gs = GridSweep(
        fn, {"x": jnp.arange(1.0, 11.0)}, batch_size=4, progress=False, workers=1
    )
    out = gs.run()
    jax.effects_barrier()
    np.testing.assert_array_equal(out, np.arange(1.0, 11.0))
    # Chunks [1-4], [5-8], [9, 10, 10, 10]: padding repeats the last point.
    expected = {float(v): 1 for v in range(1, 10)} | {10.0: 3}
    assert collections.Counter(evaluated) == expected


# ── threads: streaming, cancellation, errors ────────────────────────────────


def test_workers_1_runs_on_the_calling_thread():
    gs = GridSweep(_toy, SS, batch_size=4, progress=False, workers=1)
    threads = []
    run_chunk = gs._run_chunk

    def spy(c, device):
        threads.append(threading.current_thread())
        return run_chunk(c, device)

    gs._run_chunk = spy
    _assert_toy_grid(gs.run())
    assert set(threads) == {threading.current_thread()}


def test_several_workers_run_on_pool_threads():
    gs = GridSweep(_toy, SS, batch_size=4, progress=False, workers=4)
    names = set()
    run_chunk = gs._run_chunk

    def spy(c, device):
        names.add(threading.current_thread().name)
        return run_chunk(c, device)

    gs._run_chunk = spy
    _assert_toy_grid(gs.run())
    assert names
    assert all(name.startswith(THREAD_PREFIX) for name in names)
    assert not _sweep_threads_alive()


def test_slow_chunk_does_not_block_the_next_batch():
    # Chunk 0 (batch 0) waits until chunk 2 (batch 1) starts. A barrier at
    # each batch boundary would never start chunk 2, so the wait would time
    # out; the streaming window starts it on the other worker.
    gs = GridSweep(_toy, SS, batch_size=2, progress=False, workers=2)
    assert gs.per_worker == 1
    later_chunk_started = threading.Event()
    waited = {}
    run_chunk = gs._run_chunk

    def spy(c, device):
        if c == 2:
            later_chunk_started.set()
        if c == 0:
            waited["ok"] = later_chunk_started.wait(timeout=60)
        return run_chunk(c, device)

    gs._run_chunk = spy
    _assert_toy_grid(gs.run())
    assert waited["ok"]


def test_closing_batches_early_cancels_queued_chunks(monkeypatch):
    # Chunks after the first batch block until the pool shuts down. The
    # executor below cancels queued chunks before it releases the running
    # ones, so a queued chunk can start only if close() doesn't cancel it.
    release = threading.Event()

    class Executor(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            super().shutdown(wait=False, cancel_futures=cancel_futures)
            release.set()
            super().shutdown(wait=wait)

    monkeypatch.setattr("jbubble.utils.gridsweep.ThreadPoolExecutor", Executor)
    gs = GridSweep(
        _toy,
        {"x": jnp.arange(1000.0), "y": jnp.zeros(1)},
        batch_size=2,
        progress=False,
        workers=2,
    )
    assert gs.per_worker == 1
    lock = threading.Lock()
    started = set()
    run_chunk = gs._run_chunk

    def spy(c, device):
        with lock:
            started.add(c)
        if c >= 2:
            assert release.wait(timeout=60)
        return run_chunk(c, device)

    gs._run_chunk = spy
    it = gs.batches()
    params, _ = next(it)
    np.testing.assert_array_equal(params["x"], [0.0, 1.0])
    it.close()
    # close() joins the pool, so no thread is left to start another chunk.
    assert not _sweep_threads_alive()
    # Chunks 0 and 1 made the first batch, and each worker was running at
    # most one more chunk. The rest of the window never started.
    assert {0, 1} <= started <= {0, 1, 2, 3}


def test_window_holds_at_most_twice_workers_chunks(monkeypatch):
    # Host memory stays O(batch_size) only if the window is bounded.
    submitted = []

    class Executor(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            submitted.append(args[0])
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr("jbubble.utils.gridsweep.ThreadPoolExecutor", Executor)
    gs = GridSweep(
        _toy,
        {"x": jnp.arange(1000.0), "y": jnp.zeros(1)},
        batch_size=2,
        progress=False,
        workers=2,
    )
    it = gs.batches()
    next(it)  # consumes chunks 0 and 1
    # The two consumed chunks, plus at most 2 * workers in the window.
    assert len(submitted) <= 2 + 2 * gs.workers
    it.close()


@pytest.mark.parametrize("progress", [True, False])
def test_progress_bar_counts_grid_points(monkeypatch, progress):
    bars = []

    class Bar:
        def __init__(self, *, total, desc, unit, disable):
            self.total, self.disable, self.n = total, disable, 0
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, n):
            self.n += n

        def set_description_str(self, desc):
            pass

    monkeypatch.setattr("jbubble.utils.gridsweep.tqdm", Bar)
    gs = GridSweep(_toy, SS, batch_size=4, progress=progress, workers=2)
    _assert_toy_grid(gs.run())
    list(gs.batches())
    assert len(bars) == 2
    for bar in bars:
        assert bar.disable is not progress
        assert bar.total == bar.n == gs.total_points


# Runs in a child process that looks like a Jupyter kernel without ipywidgets
# to tqdm.autonotebook, for the detection in both old and new tqdm releases.
_IMPORT_IN_A_KERNEL_WITHOUT_IPYWIDGETS = """
import json, sys, types, warnings

class ZMQInteractiveShell:
    config = {"IPKernelApp": {}}

ipython = types.ModuleType("IPython")
ipython.get_ipython = ZMQInteractiveShell
sys.modules["IPython"] = ipython
sys.modules["ipykernel.zmqshell"] = types.ModuleType("ipykernel.zmqshell")
sys.modules["ipywidgets"] = None  # `import ipywidgets` raises ImportError

from tqdm import TqdmWarning

def tqdm_warnings(module):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        __import__(module)
    return [str(w.message) for w in caught if issubclass(w.category, TqdmWarning)]

found = {"jbubble.utils": tqdm_warnings("jbubble.utils")}
for name in ("tqdm.auto", "tqdm.autonotebook", "tqdm.notebook"):
    sys.modules.pop(name, None)
found["tqdm.auto"] = tqdm_warnings("tqdm.auto")
print(json.dumps(found))
"""


def test_importing_in_a_notebook_without_ipywidgets_does_not_warn():
    # Regression: importing jbubble.utils in a Jupyter kernel without
    # ipywidgets printed tqdm's "IProgress not found" warning.
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_IN_A_KERNEL_WITHOUT_IPYWIDGETS],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    found = json.loads(proc.stdout.splitlines()[-1])
    assert found["tqdm.auto"], (
        "tqdm.auto no longer warns in the fake kernel, so this test checks "
        f"nothing; update the fake for this tqdm release.\n{proc.stderr}"
    )
    assert found["jbubble.utils"] == []


def test_breaking_out_of_batches_stops_the_pool():
    gs = GridSweep(
        _toy,
        {"x": jnp.arange(1000.0), "y": jnp.arange(3.0)},
        batch_size=8,
        progress=False,
        workers=4,
    )
    for i, _ in enumerate(gs.batches()):
        if i == 2:
            break
    assert not _sweep_threads_alive()


def test_error_in_a_chunk_propagates_and_stops_the_pool():
    gs = GridSweep(
        _toy,
        {"x": jnp.arange(100.0), "y": jnp.zeros(1)},
        batch_size=4,
        progress=False,
        workers=4,
    )
    run_chunk = gs._run_chunk

    def spy(c, device):
        if c == 5:
            raise RuntimeError("chunk 5 failed")
        return run_chunk(c, device)

    gs._run_chunk = spy
    with pytest.raises(RuntimeError, match="chunk 5 failed"):
        gs.run()
    assert not _sweep_threads_alive()


# ── defaults ────────────────────────────────────────────────────────────────


def _default_workers() -> int:
    return GridSweep(_toy, {"x": jnp.arange(100.0), "y": jnp.zeros(1)}).workers


@cpu_only
def test_default_workers_use_process_cpu_count(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3, raising=False)
    assert _default_workers() == 3


@cpu_only
def test_default_workers_fall_back_to_cpu_affinity(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 5}, raising=False)
    assert _default_workers() == 2


@cpu_only
def test_default_workers_fall_back_to_cpu_count(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 5)
    assert _default_workers() == 5


@cpu_only
def test_default_device_on_cpu_is_the_first_local_device(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 8, raising=False)
    gs = GridSweep(_toy, SS)
    assert gs.devices == jax.local_devices()[:1]


# ── at most 31 chunks per CPU device ────────────────────────────────────────

# XLA's CPU client runs at most 32 computations at a time per device. A host
# callback that runs a JAX operation needs a free slot, so 32 chunks waiting
# in such callbacks would deadlock. The child-process tests below check that
# the sweep finishes; these check the limit itself.


@cpu_only
def test_default_workers_stop_at_31_per_cpu_device(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 64, raising=False)
    ss = {"x": jnp.arange(1000.0), "y": jnp.zeros(1)}
    assert GridSweep(_toy, ss, devices=1).workers == MAX_PER_CPU_DEVICE
    # devices=None uses one CPU device per 31 workers, as far as they go.
    gs = GridSweep(_toy, ss)
    n_local = jax.local_device_count()
    assert gs.workers == min(64, MAX_PER_CPU_DEVICE * n_local)
    assert (
        gs.devices == jax.local_devices()[: math.ceil(gs.workers / MAX_PER_CPU_DEVICE)]
    )


@cpu_only
def test_explicit_workers_above_31_per_cpu_device_warn_and_stop_at_31():
    ss = {"x": jnp.arange(1000.0), "y": jnp.zeros(1)}
    with pytest.warns(UserWarning, match="at most 31 chunks at a time") as record:
        gs = GridSweep(_toy, ss, devices=1, workers=64)
    assert "devices=3" in str(record[0].message)
    assert record[0].filename == __file__
    assert gs.workers == MAX_PER_CPU_DEVICE
    # With devices=None, more CPU devices need JAX_NUM_CPU_DEVICES, not an
    # argument.
    n_local = jax.local_device_count()
    with pytest.warns(UserWarning, match="at most") as record:
        gs = GridSweep(_toy, ss, workers=MAX_PER_CPU_DEVICE * n_local + 9)
    message = str(record[0].message)
    assert f"set `JAX_NUM_CPU_DEVICES={n_local + 1}` before you import JAX." in message
    assert "devices=" not in message
    assert gs.workers == MAX_PER_CPU_DEVICE * n_local
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert GridSweep(_toy, ss, devices=1, workers=31).workers == 31


@cpu_only
def test_at_most_31_chunks_run_at_once_on_a_cpu_device(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 64, raising=False)
    gs = GridSweep(
        _toy,
        {"x": jnp.arange(4.0 * MAX_PER_CPU_DEVICE), "y": jnp.arange(2.0)},
        progress=False,
        devices=1,
    )
    assert gs.workers == MAX_PER_CPU_DEVICE
    peak = _peak_chunks_per_device(gs)
    # Every worker was busy at once, and no more than that.
    assert peak == {gs.devices[0]: MAX_PER_CPU_DEVICE}


def _peak_chunks_per_device(gs: GridSweep) -> dict[jax.Device, int]:
    """Run `gs` and return the peak number of chunks running on each device.

    Each chunk waits until every worker runs a chunk, so the peak reaches
    the number of tokens that the sweep hands out.
    """
    lock = threading.Lock()
    running: collections.Counter = collections.Counter()
    peak: collections.Counter = collections.Counter()
    all_busy = threading.Event()
    run_chunk = gs._run_chunk

    def spy(c, device):
        with lock:
            running[device] += 1
            peak[device] = max(peak[device], running[device])
            if running.total() >= gs.workers:
                all_busy.set()
        try:
            all_busy.wait(timeout=30)
            return run_chunk(c, device)
        finally:
            with lock:
                running[device] -= 1

    gs._run_chunk = spy
    gs.run()
    assert all_busy.is_set()
    return dict(peak)


# ── validation ──────────────────────────────────────────────────────────────


def test_invalid_search_space():
    with pytest.raises(ValueError, match="1-D"):
        GridSweep(_toy, {"x": jnp.zeros((2, 2)), "y": jnp.arange(3.0)})
    with pytest.raises(ValueError, match="non-empty"):
        GridSweep(_toy, {"x": jnp.zeros(0), "y": jnp.arange(3.0)})
    with pytest.raises(ValueError, match="at least one axis"):
        GridSweep(_toy, {})
    with pytest.raises(TypeError, match="strings"):
        GridSweep(_toy, {1: jnp.arange(3.0)})


def test_invalid_sizes():
    with pytest.raises(ValueError, match="batch_size"):
        GridSweep(_toy, SS, batch_size=0)
    with pytest.raises(ValueError, match="workers"):
        GridSweep(_toy, SS, workers=0)
    with pytest.raises(TypeError):
        GridSweep(_toy, SS, batch_size=2.5)


def test_invalid_devices():
    n_local = jax.local_device_count()
    with pytest.raises(ValueError, match="set `workers` instead"):
        GridSweep(_toy, SS, devices=n_local + 1)
    with pytest.raises(ValueError):
        GridSweep(_toy, SS, devices=0)
    with pytest.raises(ValueError, match="empty"):
        GridSweep(_toy, SS, devices=[])
    with pytest.raises(ValueError, match="twice"):
        GridSweep(_toy, SS, devices=[jax.local_devices()[0]] * 2)
    with pytest.raises(TypeError):
        GridSweep(_toy, SS, devices=["cpu"])
    with pytest.raises(TypeError):
        GridSweep(_toy, SS, devices=True)


def test_fn_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        GridSweep("not a function", SS)


def test_parallel_argument_is_gone():
    with pytest.raises(TypeError, match="parallel"):
        GridSweep(_toy, SS, parallel=True)


# ── simulations ─────────────────────────────────────────────────────────────


def test_implicit_solver_runs_on_worker_threads():
    # Regression: implicit diffrax solvers (lineax linear solves) failed under
    # the old pmap path with "pytree does not match out_structure".
    ks = jnp.linspace(1.0, 50.0, 9)
    out = GridSweep(
        _decay_kvaerno5, {"k": ks}, batch_size=4, progress=False, workers=4
    ).run()
    np.testing.assert_allclose(out, np.exp(-np.asarray(ks)), rtol=1e-5, atol=1e-9)


def test_kellermiksis_kvaerno5_threads_bitwise_equal_serial():
    cfg = SolverConfig(solver=diffrax.Kvaerno5())

    def fn(R0, pressure):
        eom, pulse = _keller_miksis(R0, pressure)
        result = run_simulation(eom, pulse, save_spec=SaveSpec(32), config=cfg)
        return {"R": result.radius, "ok": result.converged}

    ss = {"R0": jnp.linspace(2e-6, 4e-6, 3), "pressure": jnp.array([50e3, 150e3])}
    threads = GridSweep(fn, ss, batch_size=6, progress=False, workers=3)
    serial = GridSweep(fn, ss, batch_size=2, progress=False, workers=1)
    # Same chunk shape, so both run the same compiled program.
    assert threads.per_worker == serial.per_worker == 2
    assert threads.workers == 3
    a, b = threads.run(), serial.run()
    assert a["R"].shape == (3, 2, 32)
    assert a["ok"].all()
    assert np.isfinite(a["R"]).all()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_nonconverged_points_are_flagged_not_fatal():
    cfg = SolverConfig(max_steps=50)  # far too few steps: every solve fails

    def fn(R0):
        eom, pulse = _keller_miksis(R0, 100e3)
        return run_simulation(eom, pulse, save_spec=SaveSpec(8), config=cfg)

    out = GridSweep(
        fn, {"R0": jnp.linspace(1e-6, 5e-6, 5)}, progress=False, workers=2
    ).run()
    assert out.converged.shape == (5,)
    assert out.converged.dtype == bool
    assert not out.converged.any()
    assert out.radius.shape == (5, 8)


# ── several devices (child process, never skipped) ──────────────────────────

N_CHILD_DEVICES = 4


def _assert_device_count(n_devices: int) -> None:
    # Runs in the child process: fail loudly if the device count didn't apply.
    assert jax.local_device_count() == n_devices, (
        f"expected {n_devices} CPU devices from JAX_NUM_CPU_DEVICES, "
        f"got {jax.local_devices()}"
    )


def _check_grid_on_several_devices(n_devices: int) -> None:
    _assert_device_count(n_devices)
    local = jax.local_devices()
    for devices in (n_devices, local[::-1], local[:2]):
        for workers in (n_devices, 2 * n_devices + 1):
            for batch_size in (1, 7, 512):
                gs = GridSweep(
                    _toy,
                    SS,
                    batch_size=batch_size,
                    progress=False,
                    devices=devices,
                    workers=workers,
                )
                _assert_toy_grid(gs.run())

    # One trace, one executable per device, and every device runs chunks.
    traces = []

    def fn(x):
        traces.append(1)
        return x**2

    gs = GridSweep(
        fn,
        {"x": jnp.arange(64.0)},
        batch_size=8,
        progress=False,
        devices=n_devices,
        workers=n_devices,
    )
    used = set()
    run_chunk = gs._run_chunk

    def spy(c, device):
        used.add(device)
        return run_chunk(c, device)

    gs._run_chunk = spy
    np.testing.assert_array_equal(gs.run(), np.arange(64.0) ** 2)
    gs.run()
    assert len(traces) == 1
    assert set(gs._executables) == set(local)
    assert used == set(local)

    # A grid with fewer chunks than devices compiles only for the ones it uses.
    small = GridSweep(fn, {"x": jnp.arange(3.0)}, progress=False, devices=n_devices)
    assert small.devices == local[:3]

    try:
        GridSweep(_toy, SS, devices=n_devices, workers=n_devices - 1)
    except ValueError as err:
        assert "each device needs at least one worker" in str(err)
    else:
        raise AssertionError("workers < devices should raise ValueError")


def _check_implicit_solver_on_several_devices(n_devices: int) -> None:
    _assert_device_count(n_devices)
    ks = jnp.linspace(1.0, 50.0, 2 * n_devices + 1)
    out = GridSweep(
        _decay_kvaerno5,
        {"k": ks},
        batch_size=n_devices,
        progress=False,
        devices=n_devices,
    ).run()
    np.testing.assert_allclose(out, np.exp(-np.asarray(ks)), rtol=1e-5, atol=1e-9)


def _check_devices_bitwise_equal_serial(n_devices: int) -> None:
    _assert_device_count(n_devices)

    def fn(R0, pressure):
        eom, pulse = _keller_miksis(R0, pressure)
        result = run_simulation(eom, pulse, save_spec=SaveSpec(32))
        return {"R": result.radius, "ok": result.converged}

    ss = {"R0": jnp.linspace(1e-6, 5e-6, 3), "pressure": jnp.linspace(50e3, 3e5, 3)}
    many = GridSweep(
        fn,
        ss,
        batch_size=2 * n_devices,
        progress=False,
        devices=n_devices,
        workers=n_devices,
    )
    serial = GridSweep(fn, ss, batch_size=2, progress=False, workers=1)
    assert many.per_worker == serial.per_worker == 2
    a, b = many.run(), serial.run()
    assert a["ok"].all()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def _check_host_callback_with_jax_op_finishes(n_devices: int) -> None:
    # Regression: with 32 or more chunks on one CPU device, every computation
    # slot belonged to a chunk waiting in this callback, and the JAX
    # operation inside the callback waited for a free slot forever.
    _assert_device_count(n_devices)

    def check(v):
        if v < 0:  # compares a jax.Array, so it runs a JAX computation
            raise ValueError("negative")

    def fn(x):
        jax.debug.callback(check, x)
        return 2 * x

    # Four points per worker, so every worker runs a chunk at the limit.
    x = jnp.arange(4.0 * MAX_PER_CPU_DEVICE * n_devices)
    # A machine with 64 cores per device: the default stops at 31 per device.
    os.process_cpu_count = lambda: 64 * n_devices
    gs = GridSweep(fn, {"x": x}, batch_size=x.size, progress=False)
    np.testing.assert_array_equal(gs.run(), 2 * np.asarray(x))
    assert len(gs.devices) == n_devices
    assert gs.workers == MAX_PER_CPU_DEVICE * n_devices

    # An explicit worker count above the limit warns and stops at it.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        gs = GridSweep(
            fn,
            {"x": x},
            batch_size=x.size,
            progress=False,
            devices=n_devices,
            workers=64 * n_devices,
        )
    np.testing.assert_array_equal(gs.run(), 2 * np.asarray(x))
    assert any("at most" in str(w.message) for w in record)
    assert gs.workers == MAX_PER_CPU_DEVICE * n_devices


def _check_cpu_devices_follow_workers(n_devices: int) -> None:
    _assert_device_count(n_devices)
    local = jax.local_devices()
    ss = {"x": jnp.arange(8.0 * MAX_PER_CPU_DEVICE * n_devices)}
    # Up to 31 workers share the default CPU device.
    os.process_cpu_count = lambda: MAX_PER_CPU_DEVICE
    assert GridSweep(_toy_1d, ss).devices == local[:1]
    # Above 31 workers, devices=None adds one CPU device per 31 workers.
    os.process_cpu_count = lambda: MAX_PER_CPU_DEVICE + 1
    assert GridSweep(_toy_1d, ss).devices == local[:2]
    os.process_cpu_count = lambda: 1000
    gs = GridSweep(_toy_1d, ss, progress=False)
    assert gs.devices == local
    assert gs.workers == MAX_PER_CPU_DEVICE * n_devices
    # Every device runs its share of the workers at once, and no more.
    peak = _peak_chunks_per_device(gs)
    assert peak == dict.fromkeys(local, MAX_PER_CPU_DEVICE)


def _toy_1d(x):
    return x**2


@pytest.mark.parametrize(
    ("check", "n_devices"),
    [
        ("_check_grid_on_several_devices", N_CHILD_DEVICES),
        ("_check_implicit_solver_on_several_devices", N_CHILD_DEVICES),
        ("_check_devices_bitwise_equal_serial", N_CHILD_DEVICES),
        ("_check_host_callback_with_jax_op_finishes", 1),
        ("_check_host_callback_with_jax_op_finishes", N_CHILD_DEVICES),
        ("_check_cpu_devices_follow_workers", N_CHILD_DEVICES),
    ],
)
def test_in_child_process(check, n_devices):
    env = dict(os.environ)
    env["JAX_NUM_CPU_DEVICES"] = str(n_devices)
    env["JAX_PLATFORMS"] = "cpu"
    # A device count in XLA_FLAGS would conflict with JAX_NUM_CPU_DEVICES.
    env["XLA_FLAGS"] = " ".join(
        flag
        for flag in env.get("XLA_FLAGS", "").split()
        if "xla_force_host_platform_device_count" not in flag
    )
    # On a hang, the child dumps its stacks and exits before the parent's
    # timeout kills it.
    code = (
        "import faulthandler, runpy; faulthandler.dump_traceback_later(240, exit=True); "
        f"runpy.run_path({__file__!r})[{check!r}]({n_devices})"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=270,
        )
    except subprocess.TimeoutExpired as err:
        pytest.fail(f"{check} hung with {n_devices} CPU device(s):\n{err.stderr}")
    assert proc.returncode == 0, (
        f"{check} failed with {n_devices} CPU device(s):\n{proc.stdout}\n{proc.stderr}"
    )
