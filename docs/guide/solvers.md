# Solvers and stiffness

[`run_simulation`][jbubble.simulation.run_simulation] integrates the
equation of motion with [diffrax](https://docs.kidger.site/diffrax/), using
the settings in a [`SolverConfig`][jbubble.solver.SolverConfig]. The defaults
suit microbubbles in water, including inertial collapse. This page explains
what the settings mean, how to tell when a solve failed, when to switch to
an implicit solver, and how gradients flow through a solve.

Every code block on this page runs as written, in order.

## The default solver

`SolverConfig()` uses the following settings:

| Setting | Default | Meaning |
|---|---|---|
| `solver` | `diffrax.Dopri5()` | Explicit fifth-order Runge-Kutta method |
| `stepsize_controller` | `diffrax.PIDController(rtol=1e-6, atol=1e-10)` | Adaptive step size with these tolerances |
| `dt0` | `1e-9` | First step size [s] |
| `max_steps` | `100_000` | Step limit per solve, accepted and rejected steps together |

`Dopri5` needs about 60 to 100 steps per driving period for a microbubble in
water, so the step limit covers about a thousand periods. To pass other
settings, give `run_simulation`, [`solve_eom`][jbubble.solver.solve_eom], or
[`fit_parameters`][jbubble.fitting.fit_parameters] a `config`:

```python
import diffrax
import jax.numpy as jnp

from jbubble import SolverConfig, run_simulation, solve_eom
from jbubble.utils.presets import lipid_bubble

eom, pulse = lipid_bubble(R0=2e-6, freq=1e6, pressure=100e3)
precise = SolverConfig(
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-12)
)
result = run_simulation(eom, pulse, config=precise)
```

## What the tolerances mean

The solver doesn't integrate the state in SI units. It integrates a scaled
state: the radius in units of $R_0$ and the wall velocity in units of
$\sqrt{P_\text{amb}/\rho_L}$, about 10 m/s in water at atmospheric pressure.
The tolerances therefore mean the same thing for a 50 nm bubble as for a
50 µm one: `atol=1e-10` is $10^{-10} R_0$ for the radius, and about
$10^{-9}$ m/s for the wall velocity.

At the defaults, a microbubble driven at 100 kPa has radius errors of
$10^{-6}$ to $10^{-5} R_0$, and gradients with relative errors of $10^{-5}$
to $10^{-3}$. Two cases are harder:

- The corners of a lipid shell's surface tension law, where the shell
  buckles and ruptures: radius errors up to about $10^{-3} R_0$, and
  gradient errors of a few percent, or about 12 % for the peak radius of
  the `lipid_bubble` preset (see [Gradients and adjoints](#gradients-and-adjoints)).
- An inertial collapse, with $R_\text{max}/R_0 \approx 4$: radius and
  gradient errors of about $10^{-3}$ to $10^{-2}$.

`rtol=1e-8, atol=1e-12`, as in the `precise` configuration, makes these
errors 100 to 1000 times smaller for two to three times as many steps. Don't
loosen the tolerances for gradients through a collapse: they can then be
wrong by more than 100 %.

`rtol` sets the accuracy of most solves. `atol` matters only where a scaled
state component is far below 1, such as the wall velocity of a small or
weakly driven bubble.

## Check that a solve converged

When a solve fails, for example because it reaches `max_steps`,
`result.converged` is `False`, and the samples after the failure are `inf`.
Called directly, `run_simulation` also warns. Under `jax.jit` or `jax.vmap`,
it can't inspect the flag, so it doesn't warn; check `result.converged`
yourself.

The following code sets a step limit that's too small for a five-cycle
pulse:

```{.python continuation}
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # run_simulation warns that it failed
    short = run_simulation(eom, pulse, config=SolverConfig(max_steps=200))

print("converged:", bool(short.converged))
print("inf samples:", int(jnp.isinf(short.radius).sum()), "of", short.radius.size)
```

To fix a solve that reaches `max_steps`, raise the limit for a long pulse,
or switch to [`SolverConfig.stiff`][jbubble.solver.SolverConfig.stiff] if
`Dopri5` needs thousands of steps per driving period. A larger `max_steps`
costs neither memory nor compile time.

## When stiffness matters

A bubble is stiff when its fastest relaxation rate, set by viscous damping,
is much larger than the driving angular frequency $\omega$. For a liquid of
viscosity $\mu$ and a lipid shell of surface viscosity $\kappa_s$, that rate
is about

$$
\lambda = \frac{4\mu + 4\kappa_s/R_0}{\rho_L R_0^2},
$$

divided by $1 + (4\mu + 4\kappa_s/R_0)/(\rho_L c_L R_0)$ for Keller-Miksis.
The rate grows as $1/R_0^2$ or faster, so small bubbles become stiff first.
The problem is stiff when the stiffness ratio $\lambda/\omega$ is above about
100:

```{.python continuation}
import math


def stiffness_ratio(R0, freq, mu=1e-3, kappa_s=0.0, rho_L=998.0, c_L=1500.0):
    """Ratio of the viscous relaxation rate to the driving angular frequency."""
    damping = 4 * mu + 4 * kappa_s / R0
    rate = damping / (rho_L * R0**2) / (1 + damping / (rho_L * c_L * R0))
    return rate / (2 * math.pi * freq)


print(f"2 µm lipid bubble, 1 MHz:   {stiffness_ratio(2e-6, 1e6, kappa_s=7.5e-9):6.1f}")
print(f"100 nm lipid bubble, 5 MHz: {stiffness_ratio(100e-9, 5e6, kappa_s=7.5e-9):6.1f}")
print(f"300 nm, 0.05 Pa s, 2 MHz:   {stiffness_ratio(300e-9, 2e6, mu=0.05):6.1f}")
```

A clinical microbubble is far from stiff, with a ratio below 1. Lipid-coated
nanobubbles at a few megahertz, with ratios of 150 at 150 nm, 320 at
100 nm, and 850 at 50 nm, and sub-micron bubbles in viscous liquids are
stiff.

On a stiff problem, `Dopri5` steps at the edge of its stability region. The
radius stays accurate, but the solve takes hundreds to thousands of steps per
driving period. Worse, the gradient through the solve can be wrong by orders
of magnitude, or NaN, even when the solve reports `converged = True`.

## Solve stiff problems with `SolverConfig.stiff`

[`SolverConfig.stiff()`][jbubble.solver.SolverConfig.stiff] returns settings
for the implicit `Kvaerno5` solver, with a root finder and Newton tolerances
that keep it reliable for bubbles down to 10 nm. The following code counts
the steps that each solver takes for a microbubble and a nanobubble.
[`solve_eom`][jbubble.solver.solve_eom] returns the raw `diffrax.Solution`,
whose `stats` hold the step counts:

```{.python continuation}
for R0, freq in [(2e-6, 1e6), (100e-9, 5e6)]:
    eom_r, pulse_r = lipid_bubble(R0=R0, freq=freq)
    for name, config in [("Dopri5", SolverConfig()), ("Kvaerno5", SolverConfig.stiff())]:
        solution = solve_eom(eom_r, pulse_r, config=config)
        ok = bool(diffrax.is_successful(solution.result))
        steps = int(solution.stats["num_steps"])
        print(f"R0 = {R0 * 1e9:4.0f} nm, {name:8s}: {steps:5d} steps, converged {ok}")
```

For the 2 µm bubble, both solvers take a few hundred steps. For the 100 nm
bubble, `Dopri5` takes more than 10 times as many steps as `Kvaerno5`.

An implicit step costs more than an explicit one, and `Kvaerno5` compiles
more slowly: about 2 to 6 s, against about 1 s for `Dopri5`. On a problem
that isn't stiff, it takes about as many steps as `Dopri5` but runs 5 to 10
times longer, so keep `Dopri5` unless the stiffness ratio or the step count
says otherwise.

To change the tolerances of the stiff configuration, call `stiff` with new
values, such as `SolverConfig.stiff(rtol=1e-8, atol=1e-12)`, rather than
replacing its `stepsize_controller`, so that the Newton tolerances follow.
`jax.jit`, `jax.vmap`, `jax.grad`, and
[`GridSweep`][jbubble.utils.gridsweep.GridSweep] all work with it;
`jax.pmap` doesn't.

## Gradients and adjoints

`jax.grad` differentiates through the solver. The `adjoint` argument of
`run_simulation`, `solve_eom`, and `fit_parameters` sets how:

| Adjoint | Use it for |
|---|---|
| `diffrax.RecursiveCheckpointAdjoint()` (default) | Reverse-mode gradients, such as `jax.grad` of a scalar loss. It gives the exact gradient of the discretised solution, and it checkpoints the solve so that memory stays bounded. |
| `diffrax.ForwardMode()` | Forward-mode derivatives with `jax.jacfwd` or `jax.jvp`, for a few parameters and many outputs, such as the Jacobian of a Levenberg-Marquardt fit. |

diffrax advises against `diffrax.BacksolveAdjoint`, whose gradients are
approximate.

Rejected trial steps can't poison a gradient. During a violent collapse, an
adaptive solver can try a step that reaches $R \le 0$, where the gas law is
undefined. jbubble never evaluates the equation of motion there: it reports
the step as failed, the solver rejects it, and the gradient stays finite. An
equation of motion marks its valid states with
[`is_admissible`][jbubble.bubble.eom.EquationOfMotion.is_admissible].

Gradients need tighter tolerances than radius curves where the model has
sharp features. The surface tension of a lipid shell has two corners, where
the shell buckles and ruptures, and the solver must resolve them for the
gradient to be accurate. The following code computes the derivative of the
peak radius of the [`lipid_bubble`][jbubble.utils.presets.lipid_bubble]
preset with respect to the shell viscosity, at the default and at tighter
tolerances, and compares each with a central finite difference:

```python
import diffrax
import jax

from jbubble import SolverConfig, run_simulation
from jbubble.utils.presets import lipid_bubble


def peak_ratio(kappa_s, config=None):
    eom, pulse = lipid_bubble(kappa_s=kappa_s)
    return run_simulation(eom, pulse, config=config).radius.max() / eom.R0


precise = SolverConfig(
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-12)
)
kappa_s, h = 7.5e-9, 1e-11  # [N s/m]
reference = (peak_ratio(kappa_s + h, precise) - peak_ratio(kappa_s - h, precise)) / (
    2 * h
)
for name, config in [("default", None), ("precise", precise)]:
    grad = jax.grad(peak_ratio)(kappa_s, config)
    print(f"{name:8s} jax.grad = {grad:.4e}, relative error {grad / reference - 1:+.1e}")
```

At the default tolerances the gradient is about 12 % off; at
`rtol=1e-8, atol=1e-12` it agrees with the finite difference to about
$10^{-5}$. For a free bubble, the default tolerances already give gradients
to about $10^{-5}$. Before you trust a gradient, or a fit, check it against a
finite difference at your tolerances.

For gradients on a stiff problem, use `SolverConfig.stiff()`. For 20 to
150 nm lipid bubbles at the default tolerances, the gradient of a loss on the
deviation $R/R_0 - 1$ then has a relative error of about $10^{-5}$ to
$10^{-4}$. For the full comparison of wall time and step counts
against the stiffness ratio, see the example
[Solvers and stiffness](../examples/07_solvers_and_stiffness.md).
