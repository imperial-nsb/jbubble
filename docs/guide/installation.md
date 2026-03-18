# Installation

## Prerequisites

jbubble requires Python 3.10+ and a working JAX installation. GPU support is optional but recommended for large parameter sweeps.

## Conda environment (recommended)

The project ships a conda environment spec:

```bash
conda env create -f environment.yml
conda activate bubbles
```

All development and examples assume the `bubbles` environment is active.

## Installing from source

```bash
git clone https://github.com/imperial-nsb/jbubble.git
cd jbubble
pip install -e ".[dev]"
```

The `[dev]` extra installs testing and documentation dependencies.

## Dependencies

| Package | Role |
|---|---|
| `jax` | Numerical backend, autodiff, JIT, vmap |
| `equinox` | PyTree-based neural networks and modules |
| `diffrax` | Adaptive ODE solvers (Kvaerno5) |
| `optax` | Optimisers for parameter fitting |
| `h5py` | HDF5 export/import |

## Verifying the installation

```python
import jbubble
from jbubble.utils.presets import free_bubble
import jax

preset = free_bubble()
from jbubble import run_simulation, SaveSpec
result = jax.jit(run_simulation)(
    preset.eom, preset.pulse,
    save_spec=SaveSpec(num_samples=500),
    t_max=10e-6,
)
print("converged:", bool(result.converged))
print("peak R/R0:", float(result.radius.max() / preset.eom.R0))
```

Expected output (values are approximate):

```
converged: True
peak R/R0: 2.3
```

## GPU / accelerator support

JAX automatically uses a GPU if one is available. No code changes are needed. For multi-GPU setups, use `jax.devices()` to select a device and `jax.device_put` to place arrays explicitly.

## Building the documentation

```bash
conda activate bubbles
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve   # live-preview at http://127.0.0.1:8000
mkdocs build   # static site in site/
```
