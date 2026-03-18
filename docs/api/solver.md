# solver

Low-level ODE integration primitives. Most users should prefer `run_simulation` from `jbubble.simulation`, which wraps these with post-processing. Use `solve_eom` directly when you need access to the raw `diffrax.Solution` object.

```python
from jbubble import solve_eom, SaveSpec, SolverConfig
```

---

::: jbubble.solver.SaveSpec

::: jbubble.solver.SolverConfig

::: jbubble.solver.solve_eom

---

## Solver choice and tolerances

The default solver is `Kvaerno5` (an implicit 5th-order Runge–Kutta method) with a PID step-size controller at relative tolerance $10^{-3}$ and absolute tolerance $10^{-6}$. This is appropriate for most bubble dynamics simulations.

For highly stiff problems (e.g. very small bubbles, extreme driving pressures, or large shear moduli in the medium), tighten the tolerances:

```python
import diffrax
from jbubble import SolverConfig

config = SolverConfig(
    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
    max_steps=50_000,
)
```

For gradient-based fitting, consider using the `RecursiveCheckpointAdjoint` to reduce memory usage during backpropagation:

```python
import diffrax

result = fit_parameters(
    ...,
    adjoint=diffrax.RecursiveCheckpointAdjoint(),
)
```
