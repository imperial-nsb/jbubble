# Units & scaling

## Why non-dimensionalise?

Bubble ODE systems mix length scales of order 1 µm, time scales of order 1 µs,
pressures of order 1 kPa, and masses of order 10⁻¹⁵ kg. When these quantities
are expressed in SI units, the ODE right-hand side contains numbers spanning
many orders of magnitude. This makes adaptive step-size control unreliable and
can cause catastrophic cancellation in floating-point arithmetic.

Non-dimensionalisation maps all variables to values of order 1, improving the
conditioning of the system and the reliability of the numerical integrator.

## Default scales

jbubble uses a `Units` class to define the reference scales. The defaults are:

| Quantity | Symbol | Default value |
|---|---|---|
| Length | `L_scale` | 1 µm = 1 × 10⁻⁶ m |
| Time | `T_scale` | 1 µs = 1 × 10⁻⁶ s |
| Mass | `M_scale` | 1 × 10⁻¹⁵ kg |

All derived scales follow from these three:

| Derived quantity | Scale | Value |
|---|---|---|
| Pressure | `P_scale = M_scale / (L_scale * T_scale²)` | ~1 kPa |
| Velocity | `vel_scale = L_scale / T_scale` | 1 m s⁻¹ |
| Density | `rho_scale = M_scale / L_scale³` | 1 kg m⁻³ |
| Surface tension | `sigma_scale = M_scale / T_scale²` | 1 mN m⁻¹ |

## Usage

```python
from jbubble import Units

units = Units()   # default scales

# Access individual scale factors
print(units.L_scale)    # 1e-6
print(units.T_scale)    # 1e-6
print(units.P_scale)    # ~1000 Pa
print(units.vel_scale)  # 1.0 m/s
```

`Units` is passed to `run_simulation` and to the bubble model internals. The
solver operates entirely in dimensionless variables; `SimulationResult` converts
back to SI when returning `radius`, `ts`, and `radiated_pressure`.

## Custom scales

You can override any of the three primary scales:

```python
units = Units(L_scale=1e-5, T_scale=1e-5, M_scale=1e-12)
```

!!! warning
    Changing the scales affects all derived quantities. Unless you have a
    specific reason to deviate, the defaults (`L=1µm, T=1µs`) are well-suited
    for typical contrast-agent microbubbles (1–10 µm radius, 1–20 MHz driving).
