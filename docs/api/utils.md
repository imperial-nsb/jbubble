# utils

Utility modules: quick-start presets, batched parameter sweeps, and HDF5 I/O.

---

## Presets

Quick-start factory functions for common bubble configurations.

```python
from jbubble.utils.presets import free_bubble, lipid_bubble, thick_shell_bubble
```

::: jbubble.utils.presets.BubblePreset

::: jbubble.utils.presets.free_bubble

::: jbubble.utils.presets.lipid_bubble

::: jbubble.utils.presets.thick_shell_bubble

---

## GridSweep

Memory-efficient batched Cartesian-product parameter sweeps.

```python
from jbubble.utils.gridsweep import GridSweep
```

::: jbubble.utils.gridsweep.GridSweep

### Example

```python
import jax.numpy as jnp
from jbubble.utils.gridsweep import GridSweep
from jbubble import run_simulation, SaveSpec
from jbubble.utils.presets import free_bubble

def peak_ratio(R0, pressure):
    preset = free_bubble(R0=R0, pressure=pressure)
    result = run_simulation(
        preset.eom, preset.pulse,
        save_spec=SaveSpec(500),
        t_max=10e-6,
    )
    return result.radius.max() / R0

sweep = GridSweep(
    fn=peak_ratio,
    search_space={
        "R0":       jnp.linspace(1e-6, 5e-6, 20),
        "pressure": jnp.array([50e3, 100e3, 200e3, 400e3]),
    },
    batch_size=256,
)

grid = sweep.run()         # shape (20, 4)
print(sweep.grid_shape)    # (20, 4)
print(sweep.total_points)  # 80
```

---

## HDF5 I/O

```python
from jbubble.utils.io import export_hdf5, load_hdf5
```

::: jbubble.utils.io.export_hdf5

::: jbubble.utils.io.load_hdf5

### Example

```python
from jbubble.utils.io import export_hdf5, load_hdf5
import jax.numpy as jnp

export_hdf5(
    "results.h5",
    metadata={"description": "sweep", "freq": 1e6, "n_cycles": 5},
    R0=jnp.linspace(1e-6, 5e-6, 20),
    peak_expansion=grid,
)

arrays, meta = load_hdf5("results.h5")
print(meta["description"])          # "sweep"
print(arrays["peak_expansion"].shape)  # (20, 4)
```
