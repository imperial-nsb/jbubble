# Installation

## Requirements

- Python 3.9 or later
- A working JAX installation (CPU or GPU)

## Install from PyPI

Once jbubble is published on PyPI, install it with:

```bash
pip install jbubble
```

## Development install

To install jbubble together with the optional development dependencies (testing, linting, documentation):

```bash
pip install "jbubble[dev]"
```

Alternatively, clone the repository and install in editable mode:

```bash
git clone https://github.com/imperial-nsb/jbubble.git
cd jbubble
pip install -e ".[dev]"
```

## JAX x64 mode

JAX defaults to 32-bit floating-point arithmetic. Bubble ODE solvers require double precision for numerical stability. **jbubble enables x64 mode automatically when the package is imported** — you do not need to set `jax.config.update("jax_enable_x64", True)` yourself.

!!! note
    If you enable or disable x64 mode manually *before* importing jbubble, your setting will be overridden. Import jbubble first if you need predictable behaviour.

## GPU support

JAX GPU support is not bundled with the base jbubble install. Follow the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) to install the correct `jaxlib` wheel for your CUDA version before installing jbubble.
