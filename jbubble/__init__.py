"""jbubble: differentiable microbubble dynamics primitives."""

import jax

jax.config.update("jax_enable_x64", True)

# For convenience, expose key simulation entry points
from .fitting import FitResult, fit_parameters
from .simulation import run_simulation
from .solver import SaveSpec, solve_eom

__all__ = [
    "run_simulation",
    "SaveSpec",
    "solve_eom",
    "fit_parameters",
    "FitResult",
]
