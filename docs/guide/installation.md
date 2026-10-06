# Installation

jbubble needs Python 3.12 or later. It installs JAX as a dependency, in its
CPU-only build. A GPU is optional; it speeds up large parameter sweeps (see
[Run on a GPU](#run-on-a-gpu)).

## Install from PyPI

To install jbubble with pip, run the following command:

```bash
pip install jbubble
```

To add jbubble to a [uv](https://docs.astral.sh/uv/) project, run the
following command:

```bash
uv add jbubble
```

## Install the optional extra

The `examples` extra installs `matplotlib`, which the
[example scripts](../examples/index.md) and the `jbubble.style` figure styles
need. To install jbubble with it, run one of the following commands:

```bash
pip install "jbubble[examples]"
uv add "jbubble[examples]"
```

## Install in a conda environment

To use jbubble from conda, create an environment with Python and pip, then
install jbubble with pip:

```bash
conda create -n jbubble python=3.13 pip
conda activate jbubble
pip install "jbubble[examples]"
```

## Install from source

To work on jbubble itself, clone the repository and let uv create the
environment:

```bash
git clone https://github.com/imperial-nsb/jbubble.git
cd jbubble
uv sync
```

`uv sync` creates a virtual environment in `.venv/` with jbubble, both
extras, and the development tools. For the full development setup, including
how to run the tests and build these docs, see the
[contributing guide](https://github.com/imperial-nsb/jbubble/blob/main/CONTRIBUTING.md).

## Check the installation

The following code runs a short simulation of a free bubble:

```python
from jbubble import run_simulation
from jbubble.utils.presets import free_bubble

eom, pulse = free_bubble()  # 2 µm air bubble, 1 MHz, 100 kPa
result = run_simulation(eom, pulse)
print("converged:", bool(result.converged))
print(f"peak R/R0: {result.radius.max() / eom.R0:.2f}")
```

It prints `converged: True` and `peak R/R0: 1.59`. The first run takes a few seconds, because JAX compiles
the solver.

## Dependencies

| Package | Role |
|---|---|
| `jax` | Arrays, compilation, vectorisation, and automatic differentiation |
| `equinox` | Models as PyTrees: every jbubble model is an `eqx.Module` |
| `diffrax` | Adaptive ODE solvers (`Dopri5` by default) |
| `optimistix` | Root finding inside the implicit solver of [`SolverConfig.stiff`][jbubble.solver.SolverConfig.stiff] |
| `lineax` | Linear solves inside diffrax's implicit solvers |
| `optax` | Optimisers for [`fit_parameters`][jbubble.fitting.fit_parameters] |
| `numpy` | Host-side arrays for sweeps |
| `tqdm` | Progress bars for [`GridSweep`][jbubble.utils.gridsweep.GridSweep] |
| `matplotlib` | Plotting in the examples (optional, `examples` extra) |

!!! note "64-bit floats"
    Importing jbubble turns on JAX's 64-bit mode
    (`jax.config.update("jax_enable_x64", True)`) for the whole process,
    because bubble dynamics need double precision. Arrays that you create
    with `jax.numpy` after the import are 64-bit by default.

## Run on a GPU

`pip install jbubble` installs the CPU-only build of JAX. To use an NVIDIA
GPU on Linux, install JAX's CUDA build after jbubble:

```bash
pip install --upgrade "jax[cuda13]"
```

For CUDA 12, use `jax[cuda12]`. In a uv project, run
`uv add "jax[cuda13]"` instead. For TPUs, other GPUs, and other platforms, see
the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

With a GPU build installed, JAX runs on the GPU automatically, and your
jbubble code doesn't change. To list the devices that JAX found, run the
following command:

```bash
python -c "import jax; print(jax.devices())"
```

A single simulation is a sequential time integration, so it's rarely faster
on a GPU. Batches of thousands of simulations are where a GPU helps; see
[Parameter sweeps](sweeps.md#cpu-and-gpu).
