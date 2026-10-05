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

For a longer explanation, with examples, see the guide [Solvers and stiffness](../guide/solvers.md).

The default, `SolverConfig()`, is `Dopri5` (an explicit fifth-order Runge–Kutta method) with a PID step-size controller at `rtol=1e-6` and `atol=1e-10`, an initial step of 1 ns, and at most 100,000 steps. It suits microbubbles in water, including inertial collapse.

The solver integrates a dimensionless state: the radius in units of $R_0$ and the wall velocity in units of $\sqrt{P_\text{amb}/\rho_L}$. So `atol=1e-10` means $10^{-10} R_0$ for a bubble of any size. In jbubble 0.1, `atol` applied in SI units.

For stiff problems, such as lipid-coated nanobubbles or sub-micron bubbles in viscous liquids, use the implicit configuration:

```python
from jbubble import SolverConfig, run_simulation

result = run_simulation(eom, pulse, config=SolverConfig.stiff())
```

The Notes of `SolverConfig` explain how to tell whether a problem is stiff.

`GridSweep` runs implicit solvers such as `Kvaerno5` on its worker threads, like explicit ones.

To tighten the tolerances, pass them to the constructor you use, for example `SolverConfig.stiff(rtol=1e-8, atol=1e-12)`, or `SolverConfig(stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-12))`. If `result.converged` is `False` because the solve reached `max_steps`, raise `max_steps`, or switch to `SolverConfig.stiff()` if `Dopri5` needs thousands of steps per driving period.

For gradients, keep the default adjoint, `diffrax.RecursiveCheckpointAdjoint()`, which differentiates the discretised solve exactly. Pass `adjoint=diffrax.ForwardMode()` to `run_simulation` or `solve_eom` for forward-mode Jacobians, for example in a Levenberg–Marquardt fit.
