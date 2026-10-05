"""Lightweight HDF5 export for simulation data.

Saves arrays and flat metadata into a single `.h5` file, for downstream
use such as machine-learning training, plotting, and analysis rather than
round-tripping Equinox module trees.

This module needs the optional `h5py` dependency. To install it, run
`pip install 'jbubble[io]'`.
"""

from __future__ import annotations

import json
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


def export_hdf5(
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    **arrays: Any,
) -> None:
    """Save named arrays and optional metadata to an HDF5 file.

    Parameters
    ----------
    path : str or Path
        Output `.h5` file path. `export_hdf5` overwrites an existing file.
    metadata : dict, optional
        JSON-serialisable metadata, stored as an attribute on the root group.
    **arrays
        Each keyword argument becomes a dataset. `np.asarray` converts the
        values to NumPy arrays.

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

    with h5py.File(path, "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=np.asarray(arr))

        if metadata is not None:
            f.attrs["metadata"] = json.dumps(metadata)


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
