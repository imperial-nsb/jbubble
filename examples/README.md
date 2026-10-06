# Examples

These 11 scripts take you from a first simulation to learning a shell law
from data. Each one is self-contained, prints a short summary of its key
numbers, and draws its figures with jbubble's Matplotlib style.

The scripts use the [Jupytext](https://jupytext.readthedocs.io/) percent
format: a `# %%` line starts a code cell, and a `# %% [markdown]` line starts
a cell of narrative text. You can run a script with `python`, step through its
cells in an editor that supports them, such as VS Code or PyCharm, or open it
as a notebook in Google Colab. The
[example gallery](https://imperial-nsb.github.io/jbubble/examples/) shows
every example with its output.

## Run an example

1. Install jbubble with the `examples` extra, which adds Matplotlib:

   ```bash
   pip install "jbubble[examples]"
   ```

   From a clone of the repository, run `uv sync --extra examples` instead.

2. Run a script:

   ```bash
   python examples/01_first_simulation.py
   ```

   In a uv project, prefix the command with `uv run`.

Each figure opens in a window, and the script continues when you close it.
To run an example without windows, for example over SSH or in continuous
integration (CI), set the environment variable `MPLBACKEND=Agg`.

## Run a quick version

The heavier examples read the environment variable `JBUBBLE_QUICK`. To shrink
their sweeps and training loops so that they finish sooner, set it
to `1`:

```bash
JBUBBLE_QUICK=1 MPLBACKEND=Agg python examples/11_learn_shell_law.py
```

The quick results are coarser, so use the default settings to reproduce the
figures in the gallery.

## All examples

Runtimes are approximate, measured on a laptop CPU with the default
settings, and include JAX compilation. The Colab notebooks run on a CPU
runtime.

| # | Example | What you learn | Runtime | Notebook |
|---|---------|----------------|---------|----------|
| 01 | [Your first bubble simulation](01_first_simulation.py) | Run a preset, build the same model from parts, and compile a simulation once with `jax.jit` | 5 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/01_first_simulation.ipynb) |
| 02 | [Driving pulses](02_driving_pulses.py) | Compare carrier shapes, combine pulses with arithmetic, and drive a bubble with a chirp and a measured trace | 5 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/02_driving_pulses.ipynb) |
| 03 | [Shells, gases, and media](03_shells_gases_media.py) | Compare free, lipid, and polymer-shelled bubbles, and swap the gas law and the surrounding medium | 10 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/03_shells_gases_media.ipynb) |
| 04 | [Equations of motion](04_equations_of_motion.py) | Compare Rayleigh-Plesset, modified Rayleigh-Plesset, Keller-Miksis, and Gilmore, and find the onset of inertial cavitation | 5 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/04_equations_of_motion.ipynb) |
| 05 | [Parameter sweeps](05_parameter_sweeps.py) | Simulate a row of bubbles with `jax.vmap`, map the response with `GridSweep`, and save and load the results with NumPy | 30 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/05_parameter_sweeps.ipynb) |
| 06 | [Acoustic emission](06_acoustic_emission.py) | Compute the pressure that a bubble radiates, its spectrum, and the drive pressure where broadband emission starts | 20 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/06_acoustic_emission.ipynb) |
| 07 | [Solvers and stiffness](07_solvers_and_stiffness.py) | Measure when small or viscous bubbles make the problem stiff, check `converged`, and switch to `SolverConfig.stiff()` | 40 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/07_solvers_and_stiffness.ipynb) |
| 08 | [Custom physics](08_custom_physics.py) | Write your own `Property` and `MediumModel`, and add a neural network with `NeuralProperty` | 5 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/08_custom_physics.ipynb) |
| 09 | [Gradients and optimisation](09_gradients_and_optimisation.py) | Differentiate a simulation with `jax.grad`, check it against finite differences, and climb to a resonance peak with Optax | 20 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/09_gradients_and_optimisation.ipynb) |
| 10 | [Fit shell parameters](10_fit_shell_parameters.py) | Recover a lipid shell's elasticity and viscosity from noisy radius curves with `fit_parameters` | 60 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/10_fit_shell_parameters.ipynb) |
| 11 | [Learn a shell law](11_learn_shell_law.py) | Train a neural network to learn a shell's surface tension law from radius curves at several pressures | 90 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/11_learn_shell_law.ipynb) |
