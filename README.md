<h1 align="center">🫧 jbubble 🫧</h1>
<p align="center"><strong>Differentiable microbubble dynamics in JAX.</strong></p>

<p align="center">
  <a href="https://github.com/imperial-nsb/jbubble/actions/workflows/ci.yml"><img src="https://github.com/imperial-nsb/jbubble/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://pypi.org/project/jbubble/"><img src="https://img.shields.io/pypi/v/jbubble.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/jbubble/"><img src="https://img.shields.io/pypi/pyversions/jbubble.svg" alt="Supported Python versions"></a>
  <a href="https://github.com/imperial-nsb/jbubble/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://imperial-nsb.github.io/jbubble/"><img src="https://img.shields.io/badge/docs-imperial--nsb.github.io-blue.svg" alt="Documentation"></a>
  <a href="https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/01_first_simulation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open the first example in Colab"></a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imperial-nsb/jbubble/main/docs/assets/readme/hero-bubble-dark.gif">
  <img src="https://raw.githubusercontent.com/imperial-nsb/jbubble/main/docs/assets/readme/hero-bubble-light.gif" alt="A microbubble, drawn to scale, grows and collapses under a five-cycle ultrasound pulse, next to its radius-time curve and the driving pressure." width="100%">
</picture>

> **Beta.** jbubble 0.2 is a beta release, and its APIs may still change
> between minor releases without a deprecation period. To use jbubble in your
> research or to contribute, get in touch by
> [opening an issue](https://github.com/imperial-nsb/jbubble/issues).

jbubble simulates and fits the radial dynamics of microbubbles driven by
ultrasound, such as the coated bubbles in ultrasound contrast agents. It's
built on [JAX](https://github.com/jax-ml/jax),
[Equinox](https://github.com/patrick-kidger/equinox), and
[diffrax](https://github.com/patrick-kidger/diffrax), so every model is a
PyTree that you can compile with `jax.jit`, batch with `jax.vmap`, and
differentiate with `jax.grad`, through the adaptive ODE solve.

jbubble is developed by the
[Noninvasive Surgery & Biopsy Laboratory](https://www.nsblab.org/) at Imperial
College London, and was first presented at the 2026 IEEE International
Ultrasonics Symposium (IUS).

## Why jbubble

- **Compiled simulations.** `jax.jit` compiles a model once. Later calls with
  new parameter values reuse the compiled solver.
- **Batched sweeps.** `jax.vmap` simulates a row of bubbles in one call, and
  `GridSweep` maps a whole parameter grid in memory-bounded chunks on every
  CPU core, or on your GPUs.
- **Gradients and fitting.** `jax.grad` gives the derivative of any output with
  respect to any model parameter. `fit_parameters` fits parameters to measured
  radius curves or emission signals with [optax](https://github.com/google-deepmind/optax),
  with bounds, scaling, and several recordings at once.
- **Composable physics.** Combine four equations of motion (Rayleigh-Plesset,
  modified Rayleigh-Plesset, Keller-Miksis, and Gilmore) with any gas, shell,
  surface tension law, and Newtonian, viscoelastic, or power-law medium.
  Presets build a free bubble, a lipid-coated bubble, or a polymer-shelled
  bubble in one line.
- **Neural components.** Replace a physical law, such as the shell's surface
  tension, with a `NeuralProperty`, or the drive with a `NeuralPulse`, and
  train it from data with the same gradients.
- **Acoustic emission.** Compute the pressure that the bubble radiates to a
  field point with an incompressible or a quasi-acoustic monopole model, and
  turn it into a spectrum.

## Install

Install jbubble from PyPI with pip:

```bash
pip install jbubble
```

Or add it to a [uv](https://docs.astral.sh/uv/) project:

```bash
uv add jbubble
```

jbubble needs Python 3.12 or later, and installs the CPU build of JAX. Two
optional extras add more:

- `io` installs h5py, for HDF5 export and import with `jbubble.utils.io`.
- `examples` installs Matplotlib, for the example scripts and the jbubble
  figure styles.

To install both extras, run `pip install "jbubble[io,examples]"`. For GPU
support, conda, or a development install, see the
[installation guide](https://imperial-nsb.github.io/jbubble/guide/installation/).

## Quick start

Simulate a lipid-coated bubble 2 µm in radius, driven by a 1 MHz, 100 kPa
tone burst:

```python
from jbubble import run_simulation
from jbubble.utils.presets import lipid_bubble

eom, pulse = lipid_bubble(R0=2e-6, freq=1e6, pressure=100e3)
result = run_simulation(eom, pulse)
print(f"Peak radius: {float(result.radius.max() / eom.R0):.2f} R0")
```

The simulation is a JAX function, so you can batch it and differentiate it:

```{.python continuation}
import jax
import jax.numpy as jnp


def peak_expansion(pressure):
    eom, pulse = lipid_bubble(R0=2e-6, freq=1e6, pressure=pressure)
    return run_simulation(eom, pulse).radius.max() / eom.R0


pressures = jnp.linspace(50e3, 200e3, 4)
print(jax.vmap(peak_expansion)(pressures))  # four bubbles in one call
print(jax.grad(peak_expansion)(100e3))  # sensitivity to pressure [1/Pa]
```

To build the same model from a gas, a shell, and a medium, read the
[quickstart guide](https://imperial-nsb.github.io/jbubble/guide/quickstart/)
or open the
[first example in Colab](https://colab.research.google.com/github/imperial-nsb/jbubble/blob/gh-pages/examples/notebooks/01_first_simulation.ipynb).

## Visual tour

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imperial-nsb/jbubble/main/docs/assets/readme/gradient-climb-dark.gif">
  <img src="https://raw.githubusercontent.com/imperial-nsb/jbubble/main/docs/assets/readme/gradient-climb-light.gif" alt="Gradient steps climb a GridSweep map of bubble response against drive frequency and bubble radius, and stop at the resonance peak." width="100%">
</picture>

`jax.grad` and optax climb a `GridSweep` response map to the resonance peak.
See [example 09: gradients and optimisation](https://imperial-nsb.github.io/jbubble/examples/09_gradients_and_optimisation/).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imperial-nsb/jbubble/main/docs/assets/readme/neural-sigma-dark.gif">
  <img src="https://raw.githubusercontent.com/imperial-nsb/jbubble/main/docs/assets/readme/neural-sigma-light.gif" alt="Over training, a neural network's surface tension curve converges on the Marmottant law, while the simulated radius curves converge on the data." width="100%">
</picture>

A neural network learns a lipid shell's surface tension law from radius
curves, against the true Marmottant law.
See [example 11: learn a shell law](https://imperial-nsb.github.io/jbubble/examples/11_learn_shell_law/).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imperial-nsb/jbubble/main/docs/assets/readme/emission-spectrum-dark.png">
  <img src="https://raw.githubusercontent.com/imperial-nsb/jbubble/main/docs/assets/readme/emission-spectrum-light.png" alt="A map of the radiated pressure spectrum against drive pressure: harmonics of the drive frequency, then broadband emission above a marked threshold." width="100%">
</picture>

The spectrum of the radiated pressure against drive pressure, from harmonics
to broadband emission.
See [example 06: acoustic emission](https://imperial-nsb.github.io/jbubble/examples/06_acoustic_emission/).

## Documentation

The documentation lives at
**[imperial-nsb.github.io/jbubble](https://imperial-nsb.github.io/jbubble/)**:

- The [example gallery](https://imperial-nsb.github.io/jbubble/examples/) has
  11 runnable examples, from a first simulation to learning a shell law, each
  with an Open in Colab button.
- The guide covers
  [installation](https://imperial-nsb.github.io/jbubble/guide/installation/),
  a [quickstart](https://imperial-nsb.github.io/jbubble/guide/quickstart/),
  [bubble models](https://imperial-nsb.github.io/jbubble/guide/bubble_models/),
  [pulse shapes](https://imperial-nsb.github.io/jbubble/guide/pulse_shapes/),
  [fitting model parameters to data](https://imperial-nsb.github.io/jbubble/guide/fitting/),
  and [JAX tips](https://imperial-nsb.github.io/jbubble/guide/jax_tips/).
- The [API reference](https://imperial-nsb.github.io/jbubble/api/) documents
  every public class and function.

For the changes in each release, see the
[changelog](https://github.com/imperial-nsb/jbubble/blob/main/CHANGELOG.md).
For planned features, see the
[roadmap](https://github.com/imperial-nsb/jbubble/blob/main/ROADMAP.md).

## Citing jbubble

jbubble was first presented at the 2026 IEEE International Ultrasonics
Symposium (IUS). Until a paper is available, cite the software itself. The
[`CITATION.cff`](https://github.com/imperial-nsb/jbubble/blob/main/CITATION.cff)
file holds the citation metadata. On GitHub, select **Cite this repository**
in the repository sidebar to copy an APA or BibTeX entry.

## Contributing

Contributions are welcome. To set up a development environment, run the
checks, and open a pull request, see the
[contributing guide](https://github.com/imperial-nsb/jbubble/blob/main/CONTRIBUTING.md).
This project follows a
[code of conduct](https://github.com/imperial-nsb/jbubble/blob/main/CODE_OF_CONDUCT.md).

## License

jbubble is released under the MIT License. Copyright (c) 2026 Noninvasive
Surgery & Biopsy Laboratory. For details, see
[`LICENSE`](https://github.com/imperial-nsb/jbubble/blob/main/LICENSE).
