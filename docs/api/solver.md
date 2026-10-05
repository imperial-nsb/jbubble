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

The default solver is `Dopri5` (an explicit 5th-order Runge–Kutta method) with a PID step-size controller at relative tolerance $10^{-6}$ and absolute tolerance $10^{-9}$, an initial step of 1 ns, and at most 10,000 steps. This is appropriate for most bubble dynamics simulations.

For stiff problems (for example, very small bubbles, extreme driving pressures, or large shear moduli in the medium), use an implicit solver such as `Kvaerno5`, and allow more steps if `result.converged` is `False`:

```python
import diffrax
from jbubble import SolverConfig

config = SolverConfig(
    solver=diffrax.Kvaerno5(),
    max_steps=50_000,
)
```

`GridSweep` with `parallel=True` supports only explicit solvers, so set `parallel=False` when you sweep with an implicit solver.

For gradient-based fitting, consider using the `RecursiveCheckpointAdjoint` to reduce memory usage during backpropagation:

```python
import diffrax

result = fit_parameters(
    ...,
    adjoint=diffrax.RecursiveCheckpointAdjoint(),
)
```
