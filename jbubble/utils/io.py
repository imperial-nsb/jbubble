"""Lightweight HDF5 export for simulation data.

Saves arrays and flat metadata into a single `.h5` file, for downstream
use such as machine-learning training, plotting, and analysis rather than
round-tripping Equinox module trees.

This module needs the optional `h5py` dependency. To install it, run
`pip install 'jbubble[io]'`.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError as err:  # h5py is an optional dependency
    raise ImportError(
        "jbubble.utils.io needs h5py. Install it with: pip install 'jbubble[io]'"
    ) from err

__all__ = ["export_hdf5", "load_hdf5"]


def _json_default(obj: Any) -> Any:
    """Convert NumPy and JAX values that `json` can't serialise.

    NumPy scalars and 0-d arrays (NumPy or JAX) become Python scalars, and
    higher-dimensional arrays become nested lists.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "__array__"):
        arr = np.asarray(obj)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def export_hdf5(
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    **arrays: Any,
) -> None:
    """Save named arrays and optional metadata to an HDF5 file.

    `export_hdf5` writes a temporary file in the same directory and moves it
    to `path` only when every write succeeds. So a value that it can't save
    raises without creating `path` or changing an existing file.

    Parameters
    ----------
    path : str or Path
        Output `.h5` file path. `export_hdf5` replaces an existing file.
    metadata : dict, optional
        JSON-serialisable metadata, stored as an attribute on the root group.
        NumPy and JAX scalars become Python numbers, and arrays become
        nested lists.
    **arrays
        Each keyword argument becomes a dataset. `np.asarray` converts the
        values to NumPy arrays. To save a dict of arrays, such as the
        result of `GridSweep.run()` for an `fn` that returns a dict, unpack
        it: `export_hdf5(path, **grid)`.

    Raises
    ------
    TypeError
        If `metadata` contains a value that isn't JSON-serialisable, or an
        array value converts to a NumPy object array, as a dict does.

    Examples
    --------
    ```python
    result = run_simulation(eom, pulse, ...)
    p_em = jax.vmap(lambda r: emission(result, r))(distances)

    export_hdf5(
        "training_data.h5",
        ts=result.ts,
        R=result.state.R,
        R_dot=result.state.R_dot,
        p_emission=p_em,
        distances=distances,
        metadata={"R0": 2e-6, "freq": 1e6},
    )
    ```
    """
    path = Path(path)
    encoded = None if metadata is None else json.dumps(metadata, default=_json_default)
    datasets = {}
    for name, arr in arrays.items():
        data = np.asarray(arr)
        if data.dtype.kind == "O":
            raise TypeError(
                f"export_hdf5: {name}={type(arr).__name__} isn't an array of "
                "numbers. To save each entry of a dict as its own dataset, "
                "unpack it, as in export_hdf5(path, **grid)."
            )
        datasets[name] = data

    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with h5py.File(tmp, "x") as f:
            if encoded is not None:
                f.attrs["metadata"] = encoded
            for name, data in datasets.items():
                f.create_dataset(name, data=data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_hdf5(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load arrays and metadata from an HDF5 file that [`export_hdf5`][jbubble.utils.io.export_hdf5] wrote.

    Parameters
    ----------
    path : str or Path
        Path to the `.h5` file.

    Returns
    -------
    arrays : dict[str, np.ndarray]
        All datasets in the file, keyed by name.
    metadata : dict
        The metadata dict, or `{}` if the file stores none.
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        arrays = {name: np.asarray(ds) for name, ds in f.items()}
        raw = f.attrs.get("metadata", "{}")
        metadata = json.loads(raw)

    return arrays, metadata
