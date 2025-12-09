# jbubble

Simulating microbubble dynamics in JAX.

## Installation

### Set up Python environment

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your Python environment and dependencies.

1.  **Download and install Miniconda**
    
    Visit the [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html) and download the installer for your operating system (Windows, macOS, or Linux). Follow the installation instructions provided there.

2.  **Create a new environment**
    
    Open your terminal (or Anaconda Prompt on Windows) and create a new environment named `jbubble`. We recommend using Python 3.10 or newer.

    ```bash
    conda create -n jbubble python=3.10
    ```

3.  **Activate the environment**

    Activate the newly created environment:

    ```bash
    conda activate jbubble
    ```

### Install jbubble

Once your environment is active, navigate to the root directory of this repository (where `pyproject.toml` is located) and install the package.

To install in **editable mode** (recommended for development, so changes to the code are immediately reflected):

```bash
pip install -e .
```

You can also install it as a standard package:

```bash
pip install jbubble
```

This will automatically install all required dependencies (JAX, Diffrax, etc.).

## Running Examples

The repository includes examples to help you get started with `jbubble`.

### Python Scripts

You can run the example scripts directly from the terminal.

**Timing Demo:**
This script demonstrates the performance of the solver.

```bash
python examples/timing_demo.py
```

### Jupyter Notebook

To run the interactive introductory notebooks, you will need to install Jupyter in your environment:

```bash
pip install jupyterlab
```

Then launch Jupyter Lab:

```bash
jupyter lab
```

This will open a browser window. Navigate to the `examples` folder and open `intro.ipynb` to explore the library interactively.
