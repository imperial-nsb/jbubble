"""Tests for jbubble.utils.gridsweep — batched parameter-sweep utility."""

import jax.numpy as jnp
import pytest
from jbubble.utils import GridSweep


def _make_simple_sweep(fn=None, batch_size=4):
    if fn is None:

        def fn(x, y):
            return x + y

    return GridSweep(
        fn,
        {"x": jnp.array([1.0, 2.0]), "y": jnp.array([10.0, 20.0, 30.0])},
        batch_size=batch_size,
        progress=False,
    )


# ── metadata ──────────────────────────────────────────────────────────────────


def test_total_points():
    gs = _make_simple_sweep()
    assert gs.total_points == 6


def test_grid_shape():
    gs = _make_simple_sweep()
    assert gs.grid_shape == (2, 3)


def test_axes_keys():
    gs = _make_simple_sweep()
    assert set(gs.axes.keys()) == {"x", "y"}


# ── collect ───────────────────────────────────────────────────────────────────


def test_collect_flat_shape():
    gs = _make_simple_sweep()
    flat = gs.collect()
    assert flat.shape == (6,)


def test_collect_values_correct():
    gs = GridSweep(
        lambda x, y: x + y,
        {"x": jnp.array([1.0]), "y": jnp.array([10.0, 20.0])},
        batch_size=4,
        progress=False,
    )
    flat = gs.collect()
    # x=1, y=10 → 11; x=1, y=20 → 21
    values = sorted(float(v) for v in flat)
    assert values == pytest.approx([11.0, 21.0])


# ── reshape ───────────────────────────────────────────────────────────────────


def test_reshape_gives_grid_shape():
    gs = _make_simple_sweep()
    flat = gs.collect()
    grid = gs.reshape(flat)
    assert grid.shape == (2, 3)


def test_reshape_values_correct():
    gs = GridSweep(
        lambda x, y: x + y,
        {"x": jnp.array([1.0, 2.0]), "y": jnp.array([10.0, 20.0])},
        batch_size=4,
        progress=False,
    )
    grid = gs.reshape(gs.collect())
    # The axes are sorted alphabetically: x first, y second
    # grid[0,0] = x[0]+y[0] = 11, grid[0,1] = x[0]+y[1] = 21
    # grid[1,0] = x[1]+y[0] = 12, grid[1,1] = x[1]+y[1] = 22
    assert float(grid[0, 0]) == pytest.approx(11.0)
    assert float(grid[0, 1]) == pytest.approx(21.0)
    assert float(grid[1, 0]) == pytest.approx(12.0)
    assert float(grid[1, 1]) == pytest.approx(22.0)


# ── batches ───────────────────────────────────────────────────────────────────


def test_batches_yields_all_points():
    gs = _make_simple_sweep(batch_size=2)
    total = sum(len(next(iter(params.values()))) for params, _ in gs.batches())
    assert total == gs.total_points


def test_batches_output_shapes_match():
    gs = _make_simple_sweep(batch_size=4)
    for params, outputs in gs.batches():
        batch_n = len(next(iter(params.values())))
        assert outputs.shape[0] == batch_n
