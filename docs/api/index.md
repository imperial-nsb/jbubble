# API reference

This section provides the full autodoc reference for every public class and
function in jbubble. Each page corresponds to one submodule.

## Modules

| Module | Contents |
|---|---|
| [`jbubble.bubble`](bubble.md) | Interfaces (`EquationOfMotion`, `GasModel`, `ShellModel`, `MediumModel`, `SurfaceTensionModel`) and all concrete components |
| [`jbubble.units`](units.md) | `Units` — non-dimensionalisation scales |
| [`jbubble.pulse`](pulse.md) | `Pulse` — ultrasound pulse descriptor |
| [`jbubble.shapes`](shapes.md) | `PulseShape` base class and all concrete waveform shapes |
| [`jbubble.solver`](solver.md) | `SaveSpec`, `solve_eom` — low-level ODE solver interface |
| [`jbubble.simulation`](simulation.md) | `run_simulation`, `SimulationResult`, `compute_radius_metrics`, `arrays_from_result` |
| [`jbubble.utils`](utils.md) | `GridSweep` — Cartesian parameter sweep utility |

## Top-level imports

The following symbols are re-exported from the package root for convenience:

```python
from jbubble import (
    # Interfaces
    GasModel, SurfaceTensionModel, ShellModel, MediumModel, EquationOfMotion,
    # Gas models
    PolytropicGas, VanDerWaalsGas,
    # Surface tension models
    ConstantSigma, MarmottantSigma, GompertzSigma,
    # Shell models
    NoShell, LipidShell, ThickShell,
    # Medium models
    KelvinVoigtMedium, NeoHookeanMedium,
    # Equations of motion
    RayleighPlesset, ModifiedRayleighPlesset, KellerMiksis,
    LeightonTube, SphericalConfinement,
    # Simulation
    Units, Pulse, SaveSpec, solve_eom,
    run_simulation, SimulationResult,
    compute_radius_metrics, arrays_from_result,
)
```
