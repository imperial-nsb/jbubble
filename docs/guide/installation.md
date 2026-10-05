# Installation

## Prerequisites

jbubble requires Python 3.12 or later. pip and uv install JAX as a dependency, in its CPU-only build. A GPU is optional but speeds up large parameter sweeps; to set one up, see [GPU and accelerator support](#gpu-and-accelerator-support).

## Install from PyPI

Install jbubble with pip:

```bash
pip install jbubble
```

Or add it to a uv project:

```bash
uv add jbubble
```

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `io` | `h5py` | HDF5 export and import with `jbubble.utils.io` |
| `examples` | `matplotlib` | Plotting in the example scripts |

To install jbubble with both extras, run the following command:

```bash
pip install "jbubble[io,examples]"
```

## Install in a conda environment

To use jbubble from conda, create an environment with Python and pip, then install jbubble with pip:

```bash
conda create -n jbubble python=3.13 pip
conda activate jbubble
pip install "jbubble[io,examples]"
```

## Install from source

```bash
git clone https://github.com/imperial-nsb/jbubble.git
cd jbubble
uv sync
```

`uv sync` creates a virtual environment in `.venv/` with jbubble, both extras, and the development tools. For the full development setup, including a conda route, see the [contributing guide](https://github.com/imperial-nsb/jbubble/blob/main/CONTRIBUTING.md).

## Dependencies

| Package | Role |
|---|---|
| `jax` | Numerical backend, autodiff, JIT, vmap |
| `equinox` | PyTree-based neural networks and modules |
| `diffrax` | Adaptive ODE solvers (Dopri5 by default) |
| `lineax` | Linear solves inside diffrax's implicit solvers |
| `optax` | Optimisers for parameter fitting |
| `numpy` | Host-side arrays for HDF5 export and import |
| `tqdm` | Progress bars for `GridSweep` |
| `h5py` | HDF5 export and import (optional, `io` extra) |
| `matplotlib` | Plotting in the example scripts (optional, `examples` extra) |

## Verify the installation

```python
import jbubble
from jbubble.utils.presets import free_bubble
import jax

preset = free_bubble()
from jbubble import run_simulation, SaveSpec

result = jax.jit(run_simulation)(
    preset.eom,
    preset.pulse,
    save_spec=SaveSpec(num_samples=500),
    t_max=10e-6,
)
print("converged:", bool(result.converged))
print("peak R/R0:", float(result.radius.max() / preset.eom.R0))
```

Expected output (values are approximate):

```
converged: True
peak R/R0: 1.6
```

## GPU and accelerator support

`pip install jbubble` installs the CPU-only build of JAX. To use an NVIDIA GPU on Linux, install JAX's CUDA build after jbubble:

```bash
pip install --upgrade "jax[cuda13]"
```

For CUDA 12, use `jax[cuda12]`. In a uv project, run `uv add "jax[cuda13]"` instead. For TPUs, AMD and Intel GPUs, and other platforms, see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

With a GPU build installed, JAX uses the GPU automatically, and jbubble needs no code changes. To check which devices JAX found, run `python -c "import jax; print(jax.devices())"`. For multi-GPU setups, use `jax.devices()` to select a device and `jax.device_put` to place arrays explicitly.

## Build the documentation

From a source checkout, install the `docs` dependency group and start the live preview:

```bash
uv sync --group docs
uv run mkdocs serve   # live-preview at http://127.0.0.1:8000
uv run mkdocs build   # static site in site/
```
