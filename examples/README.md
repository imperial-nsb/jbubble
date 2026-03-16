# jbubble Examples

This directory contains clean, minimal examples of how to use the `jbubble` library. They are ordered by complexity.

## Getting Started

1. **[01_basic_simulation.py](01_basic_simulation.py)**: The simplest possible microbubble simulation using the high-level `run_simulation` API.
2. **[02_pulse_algebra.py](02_pulse_algebra.py)**: Demonstrates the powerful pulse algebra system for creating complex acoustic driving waveforms.
3. **[03_shell_models.py](03_shell_models.py)**: Shows how to use different shell models (e.g., Lipid shells with complex surface tension).
4. **[04_batch_sweeps.py](04_batch_sweeps.py)**: Demonstrates how to use JAX's `vmap` to run thousands of simulations in parallel across a parameter grid.
5. **[05_fitting.py](05_fitting.py)**: Shows how to use the differentiable nature of `jbubble` to fit physics parameters to experimental data.
6. **[06_jit_timing.py](06_jit_timing.py)**: Performance demonstration showing the speedup from JAX's Just-In-Time (JIT) compilation.
7. **[07_cavitation_regimes.py](07_cavitation_regimes.py)**: A physics-focused example comparing stable vs inertial cavitation behaviors.

## Running the Examples

Ensure you have installed `jbubble` in your environment:

```bash
pip install -e .
```

Then run any example script:

```bash
python examples/01_basic_simulation.py
```
