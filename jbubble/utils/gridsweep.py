"""
gridsweep — memory-efficient batched generator over a Cartesian product of
named parameter axes.

Two public entry points:

    run()     – run the full sweep; returns a grid-shaped PyTree
                  shape (*grid_shape, *leaf_shape)
    batches() – lazy iterator yielding ``(params, outputs)`` one vmapped batch
                  at a time so memory stays O(batch_size)

What you do with each batch is entirely up to the caller — stream to disk,
accumulate statistics, track a running argmin, build a dataset, …

Usage
-----
::

    from jbubble.utils import GridSweep
    import jax.numpy as jnp

    gs = GridSweep(run_simulation, {"R0": radii, "p0": pressures})

    # full sweep → grid-shaped PyTree
    grid = gs.run()   # shape (*grid_shape, ...)

    # or stream batch-by-batch
    for params, outputs in gs.batches():
        process(params, outputs)

Notes
-----
* ``fn`` is called via ``jax.vmap`` (single device) or ``jax.pmap`` of a
  ``jax.vmap`` (many devices); it must be JAX-compatible.  See ``parallel``.
* Grid order is row-major (last axis varies fastest), matching
  ``numpy.unravel_index`` conventions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from tqdm import tqdm

PyTree = Any


class GridSweep:
    """Batched Cartesian-product sweep over named parameter axes.

    Parameters
    ----------
    fn :
        ``fn(**params) → PyTree``
        Called with scalar keyword args (one per axis name); must be
        JAX-vmappable and jit-able.
    search_space :
        ``{param_name: 1-D array of values}``
        The Cartesian product of all axes is swept.
    batch_size :
        Number of grid points evaluated per call.
    progress :
        Show a ``tqdm`` progress bar during iteration.
    parallel :
        If ``True``, evaluate each batch with a ``jax.pmap`` of a ``jax.vmap``:
        ``vmap`` fans the batch out within each device and ``pmap`` fans it
        across all local devices.  Default ``True``.

        Only works with explicit ODE solvers (``Dopri5``, ``Tsit5``, …);
        whereas implicit solvers (``Kvaerno5`, etc.), are currently not supported
        — leave ``parallel=False`` when sweeping a stiff, implicit-solver simulation.

        On CPU, JAX exposes a single device by default, so set
        ``XLA_FLAGS="--xla_force_host_platform_device_count=N"`` *before*
        importing JAX to expose ``N`` cores as devices for ``pmap`` to use.
    """

    def __init__(
        self,
        fn: Callable,
        search_space: dict[str, jnp.ndarray],
        batch_size: int = 512,
        progress: bool = True,
        parallel: bool = True,
    ) -> None:
        self.fn = fn
        self.search_space = search_space
        self.batch_size = batch_size
        self.progress = progress

        # Stable ordering so multi-index resolution is deterministic
        self._keys = sorted(search_space)
        self._axes = [jnp.asarray(search_space[k]) for k in self._keys]
        self._sizes = [int(len(a)) for a in self._axes]
        self._N = math.prod(self._sizes)

        self._n_devices = jax.local_device_count()
        self.parallel = bool(parallel)
        if self.parallel:
            self._pmapped = jax.pmap(jax.vmap(lambda p: self.fn(**p)))
            self._eval_batch = self._eval_pmap
        else:
            self._eval_batch = jax.jit(jax.vmap(lambda p: self.fn(**p)))

    def _eval_pmap(self, params: dict) -> PyTree:
        """Evaluate a batch by sharding it across local devices with ``pmap``.

        The batch is padded up to a whole multiple of the device count so it
        reshapes to ``(n_devices, per_device)``; the padding is trimmed from
        the result, so the output is identical to the ``vmap`` path.
        """
        D = self._n_devices
        n = int(next(iter(params.values())).shape[0])
        pad = (-n) % D
        if pad:
            params = jax.tree.map(
                lambda x: jnp.concatenate(
                    [x, jnp.broadcast_to(x[-1:], (pad, *x.shape[1:]))]
                ),
                params,
            )
        per = (n + pad) // D
        sharded = jax.tree.map(lambda x: x.reshape(D, per, *x.shape[1:]), params)
        out = self._pmapped(sharded)
        return jax.tree.map(lambda y: y.reshape(D * per, *y.shape[2:])[:n], out)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the full parameter grid (one int per axis)."""
        return tuple(self._sizes)

    @property
    def total_points(self) -> int:
        """Total number of grid points."""
        return self._N

    @property
    def axes(self) -> dict[str, jnp.ndarray]:
        """Parameter axes in sweep order."""
        return dict(zip(self._keys, self._axes, strict=True))

    def batches(self):
        """Lazy iterator over the grid.

        Yields
        ------
        params : dict[str, jnp.ndarray]
            Batch of parameter vectors, one per grid point in the batch.
        outputs : PyTree
            Corresponding vmapped outputs from ``fn``.
        """
        n_batches = math.ceil(self._N / self.batch_size)
        bar = (
            tqdm(range(n_batches), desc="Grid sweep")
            if self.progress
            else range(n_batches)
        )
        for b in bar:
            start = b * self.batch_size
            end = min(start + self.batch_size, self._N)
            flat = jnp.arange(start, end)
            multi = jnp.unravel_index(flat, self._sizes)
            params = {k: self._axes[i][multi[i]] for i, k in enumerate(self._keys)}
            outputs = self._eval_batch(params)
            yield params, outputs

    def run(self) -> PyTree:
        """Run the full sweep and return a grid-shaped PyTree.

        Returns
        -------
        PyTree
            Each leaf has shape ``(*grid_shape, *leaf_shape)``, where
            ``grid_shape`` corresponds to the axes in ``search_space``
            (sorted alphabetically, row-major — last axis varies fastest).
        """
        chunks = [outputs for _, outputs in self.batches()]
        flat = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *chunks)
        shape = self.grid_shape
        return jax.tree.map(lambda x: x.reshape(*shape, *x.shape[1:]), flat)
