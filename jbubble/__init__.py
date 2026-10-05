"""jbubble: differentiable microbubble dynamics primitives."""

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
