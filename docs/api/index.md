# API reference

This section provides the full autodoc reference for every public class and
function in jbubble. Each page corresponds to one submodule.

## Modules

| Module | Contents |
|---|---|
| [`jbubble.bubble`](bubble.md) | Bubble model classes: `RayleighPlesset`, `Marmottant`, `MarmottantGompertz`, `KellerMiksisGompertz`, `KelvinVoigtGompertz`, `LeightonGompertz`, `SphericalConfinement` |
| [`jbubble.units`](units.md) | `Units` — non-dimensionalisation scales |
| [`jbubble.pulse`](pulse.md) | `Pulse` — ultrasound pulse descriptor |
| [`jbubble.shapes`](shapes.md) | `PulseShape` base class and all concrete waveform shapes |
| [`jbubble.solver`](solver.md) | `SaveSpec`, `solve_bubble` — low-level ODE solver interface |
| [`jbubble.simulation`](simulation.md) | `run_simulation`, `SimulationResult`, `compute_radius_metrics`, `arrays_from_result` |
| [`jbubble.utils`](utils.md) | `GridSweep` — Cartesian parameter sweep utility |

## Top-level imports

The following symbols are re-exported from the package root for convenience:

```python
from jbubble import (
    # Bubble models
    RayleighPlesset,
    Marmottant,
    MarmottantGompertz,
    KellerMiksisGompertz,
    KelvinVoigtGompertz,
    LeightonGompertz,
    SphericalConfinement,
    # Simulation
    Units,
    run_simulation,
    SimulationResult,
    compute_radius_metrics,
    arrays_from_result,
)
```
