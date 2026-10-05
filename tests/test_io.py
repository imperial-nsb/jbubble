"""Tests for jbubble.utils.io."""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("h5py")

from jbubble.utils.io import export_hdf5, load_hdf5  # noqa: E402


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


class TestMetadataTypes:
    def test_numpy_and_jax_scalars_round_trip(self, tmp_path):
        path = tmp_path / "test.h5"
        meta = {
            "f32": np.float32(0.5),
            "f64": np.float64(2e-6),
            "i64": np.int64(7),
            "flag": np.bool_(True),
            "jax_scalar": jnp.asarray(1e6),
            "jax_int": jnp.asarray(3),
        }
        export_hdf5(path, metadata=meta, x=jnp.array([1.0]))
        _, loaded = load_hdf5(path)

        assert loaded == {
            "f32": 0.5,
            "f64": 2e-6,
            "i64": 7,
            "flag": True,
            "jax_scalar": 1e6,
            "jax_int": 3,
        }
        assert isinstance(loaded["i64"], int)

    def test_arrays_become_lists(self, tmp_path):
        path = tmp_path / "test.h5"
        export_hdf5(
            path,
            metadata={"distances": jnp.array([1e-3, 1e-2]), "grid": np.eye(2)},
            x=jnp.array([1.0]),
        )
        _, loaded = load_hdf5(path)
        assert loaded["distances"] == [1e-3, 1e-2]
        assert loaded["grid"] == [[1.0, 0.0], [0.0, 1.0]]

    def test_unserialisable_metadata_leaves_existing_file_untouched(self, tmp_path):
        path = tmp_path / "test.h5"
        export_hdf5(path, metadata={"run": 1}, x=jnp.array([1.0]))

        with pytest.raises(TypeError, match="not JSON serializable"):
            export_hdf5(path, metadata={"bad": object()}, y=jnp.array([2.0]))

        arrays, metadata = load_hdf5(path)
        assert set(arrays) == {"x"}
        assert metadata == {"run": 1}

    def test_unserialisable_metadata_creates_no_file(self, tmp_path):
        path = tmp_path / "new.h5"
        with pytest.raises(TypeError):
            export_hdf5(path, metadata={"bad": {1, 2}}, x=jnp.array([1.0]))
        assert not path.exists()

    def test_dict_array_leaves_existing_file_untouched(self, tmp_path):
        path = tmp_path / "test.h5"
        export_hdf5(path, metadata={"R0": 2e-6}, R=jnp.ones(3))

        with pytest.raises(TypeError, match=r"grid=dict.*\*\*grid"):
            export_hdf5(path, R=jnp.ones(3), grid={"ratio": np.ones(3)})

        arrays, metadata = load_hdf5(path)
        assert set(arrays) == {"R"}
        assert metadata == {"R0": 2e-6}
        assert [p.name for p in tmp_path.iterdir()] == ["test.h5"]

    def test_failed_write_leaves_existing_file_untouched(self, tmp_path):
        # h5py can't store NumPy unicode strings, so the write fails part way.
        path = tmp_path / "test.h5"
        export_hdf5(path, metadata={"run": 1}, x=jnp.array([1.0]))

        with pytest.raises(TypeError):
            export_hdf5(path, y=jnp.array([2.0]), labels=np.array(["a", "b"]))

        arrays, metadata = load_hdf5(path)
        assert set(arrays) == {"x"}
        assert metadata == {"run": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["test.h5"]
