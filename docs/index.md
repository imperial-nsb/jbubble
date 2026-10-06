# jbubble

**Differentiable microbubble dynamics in JAX.**

jbubble simulates the radial oscillations of acoustically driven gas bubbles,
such as ultrasound contrast agents and cavitation nuclei. You assemble a model
from interchangeable parts (an equation of motion, a gas, a shell, a
surrounding medium, and a driving pulse), and jbubble solves it with
[diffrax](https://docs.kidger.site/diffrax/). Every simulation is a pure
[JAX](https://docs.jax.dev/) function, so you can compile it with `jax.jit`,
batch it with `jax.vmap`, and differentiate it with `jax.grad`.

![A microbubble drawn to scale oscillates beside its radius trace and the driving pressure](assets/readme/hero-bubble-light.gif#only-light)
![A microbubble drawn to scale oscillates beside its radius trace and the driving pressure](assets/readme/hero-bubble-dark.gif#only-dark)

## What you can do with jbubble

- **Simulate** free, lipid-coated, and polymer-shelled bubbles with the
  Rayleigh-Plesset, Keller-Miksis, and Gilmore equations, in Newtonian
  liquids or viscoelastic tissue.
- **Drive** them with tone bursts, chirps, measured waveforms, or
  learned pulses, and combine pulses with `+` and `*`.
- **Sweep** thousands of bubbles at once with `jax.vmap` or with
  [`GridSweep`][jbubble.utils.gridsweep.GridSweep], which uses every CPU core
  by default.
- **Differentiate** any output with respect to any input, including shell
  parameters, the pulse, and the weights of a neural network.
- **Fit** model parameters to measured radius curves or hydrophone signals
  with [`fit_parameters`][jbubble.fitting.fit_parameters], or learn a
  constitutive law, such as the surface tension $\sigma(R)$, with a neural
  network.

jbubble was first presented at the 2026 IEEE International Ultrasonics
Symposium (IUS). Version 0.2 is beta software: the API can still change
between minor versions, and the [changelog](changelog.md) lists every change.

## Install

jbubble needs Python 3.12 or later.

```bash
pip install jbubble                  # or: uv add jbubble
pip install "jbubble[examples]"      # optional: Matplotlib for the examples
python -c "import jbubble; print(jbubble.__version__)"
```

To run on a GPU, or to install from source, see
[Installation](guide/installation.md).

## Quick start

The following code simulates a 2 µm lipid-coated microbubble driven by a
five-cycle, 1 MHz, 100 kPa tone burst:

```python
from jbubble import run_simulation
from jbubble.utils.presets import lipid_bubble

eom, pulse = lipid_bubble(R0=2e-6, freq=1e6, pressure=100e3)
result = run_simulation(eom, pulse)
print(f"peak R/R0 = {result.radius.max() / eom.R0:.2f}")
```

`result.radius` holds the radius $R(t)$ in metres at the times in
`result.ts`. To build the same model from its parts and plot it, see the
[Quickstart](guide/quickstart.md).

## Learn more

<div class="grid cards" markdown>

- **[Getting started](guide/installation.md)**

    Install jbubble, then run, plot, and compile your first simulation in
    the [Quickstart](guide/quickstart.md).

- **[Guide](guide/bubble_models.md)**

    How the physics fits together: [bubble models](guide/bubble_models.md),
    [pulses](guide/pulse_shapes.md), [solvers](guide/solvers.md),
    [sweeps](guide/sweeps.md), [fitting](guide/fitting.md), and
    [JAX tips](guide/jax_tips.md).

- **[Examples](examples/index.md)**

    Eleven runnable examples, from a first simulation to learning a shell
    law with a neural network. Each one opens in Google Colab.

- **[API reference](api/index.md)**

    Every public class and function, with its governing equations and
    references.

</div>

To cite jbubble in a publication, see [Citing jbubble](citing.md).
