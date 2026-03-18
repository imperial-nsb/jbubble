# API reference

All public APIs are documented here with full signatures, parameters, and governing equations.

## Module layout

```
jbubble/
├── __init__.py           # run_simulation, fit_parameters, SaveSpec, SolverConfig, solve_eom, FitResult
├── simulation.py         # SimulationResult
├── solver.py             # SaveSpec, SolverConfig, solve_eom
├── metrics.py            # mse_radius, mse_emission, …
├── fitting.py            # fit_parameters, FitResult
├── bubble/
│   ├── state.py          # BubbleState, ConfinedBubbleState
│   ├── property.py       # Property, ConstantProperty, NeuralProperty, as_property
│   ├── gas.py            # GasModel, PolytropicGas, VanDerWaalsGas
│   ├── shell.py          # ShellModel, NoShell, LipidShell, ThickShell, surface tension Properties
│   ├── medium.py         # MediumModel, NewtonianMedium, KelvinVoigtMedium, NeoHookeanMedium, PowerLawMedium
│   └── eom.py            # EquationOfMotion, RayleighPlesset, KellerMiksis, Gilmore, …
├── acoustics/
│   └── emission.py       # EmissionModel, IncompressibleMonopole, QuasiAcoustic
├── pulse/
│   ├── __init__.py       # ToneBurst, SampledPulse, ChirpPulse, NeuralPulse, Scaled, Summed, Offset, envelopes
│   └── shapes.py         # PulseShape, Sine, Square, Sawtooth, Rectangular, …
└── utils/
    ├── presets.py        # BubblePreset, free_bubble, lipid_bubble, thick_shell_bubble
    ├── gridsweep.py      # GridSweep
    └── io.py             # export_hdf5, load_hdf5
```

## Import conventions

Only top-level orchestration functions are re-exported from the package root:

```python
# Top-level imports
from jbubble import run_simulation, fit_parameters, SaveSpec, SolverConfig, solve_eom, FitResult

# Subpackage imports (required for all model classes)
from jbubble.bubble.eom import KellerMiksis, RayleighPlesset, Gilmore
from jbubble.bubble.gas import PolytropicGas, VanDerWaalsGas
from jbubble.bubble.shell import NoShell, LipidShell, ThickShell
from jbubble.bubble.shell import GompertzSurfaceTension, MarmottantSurfaceTension
from jbubble.bubble.medium import NewtonianMedium, KelvinVoigtMedium, NeoHookeanMedium, PowerLawMedium
from jbubble.bubble.property import ConstantProperty, NeuralProperty
from jbubble.bubble.state import BubbleState, ConfinedBubbleState
from jbubble.acoustics import IncompressibleMonopole, QuasiAcoustic
from jbubble.pulse import ToneBurst, ChirpPulse, SampledPulse, NeuralPulse
from jbubble.pulse import SoftRectangularEnvelope, HannEnvelope, TukeyEnvelope
from jbubble.pulse.shapes import Sine, Square, Sawtooth
from jbubble.utils.presets import free_bubble, lipid_bubble, thick_shell_bubble
from jbubble.utils.gridsweep import GridSweep
from jbubble.utils.io import export_hdf5, load_hdf5
```

## Pages

| Page | Contents |
|---|---|
| [bubble](bubble.md) | State, Property, gas, shell, medium, EoM classes |
| [pulse](pulse.md) | ToneBurst, ChirpPulse, SampledPulse, NeuralPulse, envelopes, composition |
| [shapes](shapes.md) | PulseShape subclasses (Sine, Square, Rectangular, …) |
| [acoustics](acoustics.md) | EmissionModel, IncompressibleMonopole, QuasiAcoustic |
| [solver](solver.md) | SaveSpec, SolverConfig, solve_eom |
| [simulation](simulation.md) | SimulationResult, run_simulation |
| [metrics](metrics.md) | Differentiable loss functions |
| [fitting](fitting.md) | fit_parameters, FitResult |
| [utils](utils.md) | Presets, GridSweep, HDF5 I/O |
