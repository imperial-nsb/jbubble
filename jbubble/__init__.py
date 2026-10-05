"""jbubble: differentiable microbubble dynamics primitives.

The package root exports only the top-level orchestration API:
[`run_simulation`][jbubble.simulation.run_simulation],
[`solve_eom`][jbubble.solver.solve_eom],
[`SaveSpec`][jbubble.solver.SaveSpec],
[`SolverConfig`][jbubble.solver.SolverConfig],
[`fit_parameters`][jbubble.fitting.fit_parameters], and
[`FitResult`][jbubble.fitting.FitResult]. Import model classes from their
subpackages, such as `jbubble.bubble.eom` and `jbubble.pulse`.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

import jax

try:
    __version__ = _pkg_version("jbubble")
except PackageNotFoundError:  # source tree on sys.path without an install
    __version__ = "0.0.0+unknown"

jax.config.update("jax_enable_x64", True)

# For convenience, expose key simulation entry points
from .fitting import FitResult, fit_parameters
from .simulation import run_simulation
from .solver import SaveSpec, SolverConfig, solve_eom

__all__ = [
    "__version__",
    "run_simulation",
    "SaveSpec",
    "SolverConfig",
    "solve_eom",
    "fit_parameters",
    "FitResult",
]
