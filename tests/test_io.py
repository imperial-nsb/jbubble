"""Tests for jbubble.utils.io."""

import jax.numpy as jnp
import numpy as np
from jbubble.utils.io import export_hdf5, load_hdf5


class TestHdf5RoundTrip:
    def test_arrays_round_trip(self, tmp_path):
        path = tmp_path / "test.h5"
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([[1, 2], [3, 4]])

        export_hdf5(path, a=a, b=b)
        arrays, metadata = load_hdf5(path)

        assert "a" in arrays
        assert "b" in arrays
        np.testing.assert_allclose(arrays["a"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(arrays["b"], np.array([[1, 2], [3, 4]]))

    def test_metadata_round_trip(self, tmp_path):
        path = tmp_path / "test.h5"
        meta = {"R0": 2e-6, "freq": 1e6, "description": "test sweep"}

        export_hdf5(path, metadata=meta, x=jnp.array([1.0]))
        _, loaded_meta = load_hdf5(path)

        assert loaded_meta["R0"] == 2e-6
        assert loaded_meta["freq"] == 1e6
        assert loaded_meta["description"] == "test sweep"

    def test_no_metadata(self, tmp_path):
        path = tmp_path / "test.h5"
        export_hdf5(path, x=jnp.array([1.0, 2.0]))
        arrays, metadata = load_hdf5(path)

        assert "x" in arrays
        assert metadata == {}

    def test_empty_metadata(self, tmp_path):
        path = tmp_path / "test.h5"
        export_hdf5(path, metadata={}, x=jnp.array([1.0]))
        _, metadata = load_hdf5(path)
        assert metadata == {}

    def test_multiple_arrays(self, tmp_path):
        path = tmp_path / "test.h5"
        export_hdf5(
            path,
            ts=jnp.linspace(0, 1, 100),
            R=jnp.ones(100) * 2e-6,
            R_dot=jnp.zeros(100),
        )
        arrays, _ = load_hdf5(path)

        assert len(arrays) == 3
        assert arrays["ts"].shape == (100,)
        assert arrays["R"].shape == (100,)
        assert arrays["R_dot"].shape == (100,)

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "test.h5"
        export_hdf5(path, x=jnp.array([1.0]))
        export_hdf5(path, y=jnp.array([2.0]))  # overwrite

        arrays, _ = load_hdf5(path)
        assert "y" in arrays
        assert "x" not in arrays

    def test_numpy_arrays(self, tmp_path):
        path = tmp_path / "test.h5"
        export_hdf5(path, x=np.array([1.0, 2.0, 3.0]))
        arrays, _ = load_hdf5(path)
        np.testing.assert_allclose(arrays["x"], [1.0, 2.0, 3.0])
