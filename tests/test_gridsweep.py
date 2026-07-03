"""Tests for jbubble.utils.gridsweep."""

import jax.numpy as jnp
import pytest
from jbubble.utils.gridsweep import GridSweep


class TestGridSweep:
    def test_grid_shape(self):
        gs = GridSweep(
            fn=lambda x, y: x + y,
            search_space={
                "x": jnp.array([1.0, 2.0, 3.0]),
                "y": jnp.array([10.0, 20.0]),
            },
            progress=False,
        )
        assert gs.grid_shape == (3, 2)

    def test_total_points(self):
        gs = GridSweep(
            fn=lambda x, y: x + y,
            search_space={
                "x": jnp.array([1.0, 2.0, 3.0]),
                "y": jnp.array([10.0, 20.0]),
            },
            progress=False,
        )
        assert gs.total_points == 6

    def test_axes(self):
        gs = GridSweep(
            fn=lambda x, y: x + y,
            search_space={
                "x": jnp.array([1.0, 2.0, 3.0]),
                "y": jnp.array([10.0, 20.0]),
            },
            progress=False,
        )
        axes = gs.axes
        assert "x" in axes
        assert "y" in axes
        assert len(axes["x"]) == 3
        assert len(axes["y"]) == 2

    def test_run_shape(self):
        gs = GridSweep(
            fn=lambda x, y: x * y,
            search_space={
                "x": jnp.array([1.0, 2.0, 3.0]),
                "y": jnp.array([10.0, 20.0]),
            },
            batch_size=4,
            progress=False,
        )
        result = gs.run()
        assert result.shape == (3, 2)

    def test_run_values(self):
        gs = GridSweep(
            fn=lambda x, y: x + y,
            search_space={
                "x": jnp.array([1.0, 2.0]),
                "y": jnp.array([10.0, 20.0]),
            },
            progress=False,
        )
        result = gs.run()
        # Sorted alphabetically: x first, y second
        # (x=1,y=10), (x=1,y=20), (x=2,y=10), (x=2,y=20)
        assert float(result[0, 0]) == pytest.approx(11.0)
        assert float(result[0, 1]) == pytest.approx(21.0)
        assert float(result[1, 0]) == pytest.approx(12.0)
        assert float(result[1, 1]) == pytest.approx(22.0)

    def test_batches_yields_all_points(self):
        gs = GridSweep(
            fn=lambda x, y: x + y,
            search_space={
                "x": jnp.array([1.0, 2.0, 3.0]),
                "y": jnp.array([10.0, 20.0]),
            },
            batch_size=2,
            progress=False,
        )
        total = 0
        for _params, outputs in gs.batches():
            total += outputs.shape[0]
        assert total == 6

    def test_single_axis(self):
        gs = GridSweep(
            fn=lambda x: x**2,
            search_space={"x": jnp.array([1.0, 2.0, 3.0, 4.0])},
            progress=False,
        )
        result = gs.run()
        assert result.shape == (4,)
        assert float(result[2]) == pytest.approx(9.0)

    def test_vector_output(self):
        gs = GridSweep(
            fn=lambda x: jnp.array([x, x**2]),
            search_space={"x": jnp.array([1.0, 2.0, 3.0])},
            progress=False,
        )
        result = gs.run()
        assert result.shape == (3, 2)
        assert float(result[1, 0]) == pytest.approx(2.0)
        assert float(result[1, 1]) == pytest.approx(4.0)

    def test_pmap_matches_vmap(self):
        # Forcing parallel=True exercises the pmap path (padding + reshape +
        # trim); on a single device it must give exactly the vmap result.
        # Grid size (15) is deliberately not a multiple of the device count.
        ss = {"x": jnp.arange(5.0), "y": jnp.arange(3.0)}
        serial = GridSweep(lambda x, y: x * y + 1.0, ss, batch_size=4,
                           progress=False, parallel=False).run()
        parallel = GridSweep(lambda x, y: x * y + 1.0, ss, batch_size=4,
                             progress=False, parallel=True).run()
        assert parallel.shape == serial.shape == (5, 3)
        assert bool(jnp.allclose(parallel, serial))

    def test_pmap_pytree_output(self):
        # pmap path must preserve PyTree-structured outputs.
        gs = GridSweep(
            fn=lambda x, y: {"s": x + y, "p": x * y},
            search_space={"x": jnp.arange(4.0), "y": jnp.arange(3.0)},
            batch_size=5, progress=False, parallel=True,
        )
        out = gs.run()
        assert out["s"].shape == (4, 3)
        assert float(out["s"][2, 1]) == pytest.approx(3.0)
        assert float(out["p"][3, 2]) == pytest.approx(6.0)
