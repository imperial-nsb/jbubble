# jbubble Examples

This directory contains clean, minimal examples of how to use the `jbubble` library. They are ordered by complexity.

## Getting Started

0. **[00_presets.py](00_presets.py)**: The fastest way to start: pick a preset (`free_bubble`, `lipid_bubble`, or `thick_shell_bubble`) and run. Three side-by-side plots showing the effect of shell type on bubble dynamics.
1. **[01_basic_simulation.py](01_basic_simulation.py)**: The simplest possible microbubble simulation using the high-level `run_simulation` API. Shows how to assemble an EoM from gas, shell, and medium models manually.
2. **[02_pulse_algebra.py](02_pulse_algebra.py)**: Demonstrates the composable pulse algebra system — adding two tone bursts at different frequencies, scaling amplitude, and applying a windowing envelope.
3. **[03_shell_models.py](03_shell_models.py)**: Compares different shell models (no shell, lipid Marmottant, thick Church) and surface tension formulations on identical bubble/pulse parameters.
4. **[04_batch_sweeps.py](04_batch_sweeps.py)**: Uses `GridSweep` + JAX `vmap` to run thousands of simulations across a parameter grid in a single batched call.
5. **[05_fitting.py](05_fitting.py)**: Fits a shell elasticity parameter (`chi`) to synthetic noisy data via gradient descent (`fit_parameters` + `optax`). Shows convergence plot and radius overlay.
6. **[06_jit_timing.py](06_jit_timing.py)**: Benchmarks JIT compilation overhead vs steady-state throughput, illustrating how JAX's JIT pays off for repeated simulations.
7. **[07_cavitation_regimes.py](07_cavitation_regimes.py)**: Physics-focused example contrasting stable cavitation (low-pressure, near-linear) vs inertial cavitation (high-pressure collapse).
8. **[08_acoustic_emissions.py](08_acoustic_emissions.py)**: Computes radiated acoustic pressure from a solved trajectory using `IncompressibleMonopole` and `QuasiAcoustic` emission models; also shows multi-distance `vmap`.
9. **[09_custom_pulse_shapes.py](09_custom_pulse_shapes.py)**: Shows how to extend jbubble with custom Fourier pulse shapes by subclassing `FourierPulseShape`, and catalogues the built-in `Rectangular` wave variants.
10. **[10_envelopes.py](10_envelopes.py)**: Compares all built-in envelope types (`Rectangular`, `Hann`, `Tukey`, `SoftRectangular`) and plots their time derivatives to highlight which are safe for adjoint-based gradient fitting.
11. **[11_gradient_resonance.py](11_gradient_resonance.py)**: Advanced end-to-end demo: runs a 2-D parameter sweep (frequency × bubble radius) to build an expansion-ratio heatmap, then uses `fit_parameters` to follow the gradient to the resonance peak. Saves the trajectory overlay as a plot.

## Running the Examples

Ensure you have installed `jbubble` in your environment:

```bash
pip install -e .
```

Then run any example script:

```bash
python examples/01_basic_simulation.py
```

> **Note:** Example 11 (`11_gradient_resonance.py`) runs a 100×100 parameter sweep followed by a 50-step gradient descent and can take several minutes on CPU.
