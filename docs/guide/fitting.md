# Fit model parameters to data

jbubble differentiates through the ODE solve, so you can estimate model
parameters from measured bubble dynamics: shell elasticity and viscosity, the
equilibrium radius, a trigger delay, or the weights of a neural
surface-tension law. [`fit_parameters`][jbubble.fitting.fit_parameters]
minimises a loss that you define with any
[optax](https://optax.readthedocs.io) optimiser.

Every code block on this page runs as written, in order.

## Fit one parameter

The following example estimates the shell viscosity `kappa_s` of a
lipid-coated bubble from one radius trace:

```python
import jax.numpy as jnp
import optax

from jbubble import SaveSpec, fit_parameters, run_simulation
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import GompertzSurfaceTension, LipidShell
from jbubble.fitting import Parameter
from jbubble.metrics import normalised_mse_radius
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine

R0 = 2e-6  # equilibrium radius [m]
SAVE_SPEC = SaveSpec(256)
T_MAX = 8e-6  # [s]


def make_eom(kappa_s, chi=0.5):
    sigma = GompertzSurfaceTension(R_buckle_ratio=0.98, chi=chi, sigma_rupture=0.072)
    return KellerMiksis(
        gas=PolytropicGas(gamma=1.07),
        shell=LipidShell(sigma=sigma, kappa_s=kappa_s),
        medium=NewtonianMedium(mu=1e-3),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


def make_pulse(pressure):
    return ToneBurst(freq=1e6, pressure=pressure, shape=Sine(), cycle_num=5)


# Stand-in for a measured radius trace, simulated with kappa_s = 3e-9 N s/m.
measured_radius = run_simulation(
    make_eom(3e-9), make_pulse(100e3), save_spec=SAVE_SPEC, t_max=T_MAX
).radius

fit = fit_parameters(
    make_model=lambda params: (make_eom(params["kappa_s"]), make_pulse(100e3)),
    params0={"kappa_s": Parameter(1e-9, lower=0.0)},  # initial guess [N s/m]
    loss_fn=lambda result: normalised_mse_radius(result.radius, measured_radius, R0),
    optimizer=optax.adam(0.05),
    n_steps=100,
    save_spec=SAVE_SPEC,
    t_max=T_MAX,
)
print(fit.params["kappa_s"])  # about 3e-9
```

Each step runs the forward model and differentiates the loss with respect to
the parameters:

```
params -> make_model(params) -> (eom, pulse) -> run_simulation -> loss_fn -> scalar
```

The result is a [`FitResult`][jbubble.fitting.FitResult]:

- `fit.params`: the fitted values in physical units, with the structure of
  `params0`.
- `fit.loss_history`: the loss at `params0` and after each step.
- `fit.result`: the [`SimulationResult`][jbubble.simulation.SimulationResult]
  at the fitted values.
- `fit.num_rejected`, `fit.stopped_early`, and `fit.message`: what happened
  during the fit. For details, see [Handle failed solves](#handle-failed-solves).

!!! note
    The stand-in trace is simulated with the same solver settings as the
    fit, so the loss can reach zero. With real data, the loss stops at the
    noise level.

## Declare what to fit

Physical parameters span many orders of magnitude: `kappa_s` is about
$10^{-9}$ N s/m, `chi` about 0.5 N/m, and `R0` about $10^{-6}$ m. An optimiser
such as Adam steps by about the learning rate in each value's own units, so
no single learning rate suits raw values. Wrap each physical value in a
[`Parameter`][jbubble.fitting.Parameter] instead. The optimiser then updates a
coordinate of order one, and the learning rate becomes a relative step size:
`optax.adam(0.05)` changes each value by about 5 % per step at first.

| To fit | Put this in `params0` |
|---|---|
| A positive quantity, such as a viscosity or a radius | `Parameter(1e-9, lower=0.0)` |
| A quantity between two bounds | `Parameter(0.3, lower=0.0, upper=1.7)` |
| A signed quantity that starts at zero, such as a trigger delay | `Parameter(0.0, scale=1e-7)` |
| An unbounded value, scaled by its initial value | `Parameter(0.5)`, or the Python float `0.5` |
| A value that you want to switch off for now | `Parameter(2e-6, fixed=True)` |
| The weights of a neural network | the module, for example a [`NeuralProperty`][jbubble.bubble.property.NeuralProperty] |

With bounds, a parameter can't leave the open interval, so a fit never tries
a negative viscosity. Bounds are scalars: one bound applies to every element
of an array `Parameter`. `make_model` always receives physical values.

`fit_parameters` decides what to fit from the type of each leaf of `params0`:

- A `Parameter` is fitted on its scaled coordinate, unless `fixed=True`.
- A Python float, or `np.float64`, in `params0` or in a dict, list, or tuple
  inside it, is shorthand for `Parameter(value)`.
- A floating-point array, such as a network weight, is fitted in its own
  units. So is a NumPy scalar of another type, such as `np.float32`, or a 0-d
  array, for example a value loaded from an HDF5 file.
- Everything else is held fixed: integers, booleans, strings, callables, and
  Python floats inside an Equinox module.

!!! warning "Raw arrays are fitted in their own units"
    `params0=jnp.array(1e-9)` is fitted in N s/m, so `optax.adam(1e-2)`
    proposes a first step ten million times larger than the value. If the
    first step is larger than a raw value, or smaller than a millionth of it,
    `fit_parameters` warns. Use `Parameter` for physical values.

A Python float inside an Equinox module isn't an array, so the optimiser
doesn't update it. If you set one, for example `MyParams(kappa_s=1e-9, ...)`,
`fit_parameters` warns and names it. Floats that keep their field's default,
such as a pulse's `initial_time=0.0`, don't trigger the warning. To fit such a
value, wrap it in `Parameter`. To hold it fixed without the warning, use
`Parameter(value, fixed=True)`, or declare the field with
`eqx.field(static=True)`.

Optimiser transformations that read the parameters, such as the weight decay
in `optax.adamw`, act on the optimiser's coordinates, not on the physical
values. Weight decay pulls an unbounded `Parameter` toward zero, but a
lower-bounded one toward its initial value.

## Fit several recordings at once

Recordings at several driving pressures constrain shared parameters far
better than one recording does. Pass one entry per recording in
`conditions`. Each entry can be any pytree, such as a dict. `fit_parameters`
passes it to `make_model` and `loss_fn` as a second argument and averages the
loss over the recordings:

```{.python continuation}
pressures = [50e3, 100e3, 150e3]  # [Pa]
conditions = [
    {
        "pressure": p,
        "radius": run_simulation(
            make_eom(3e-9), make_pulse(p), save_spec=SAVE_SPEC, t_max=T_MAX
        ).radius,  # stand-in for the measured trace at this pressure
    }
    for p in pressures
]


def make_model(params, condition):
    eom = make_eom(params["kappa_s"], chi=params["chi"])
    return eom, make_pulse(condition["pressure"])


def loss_fn(result, condition):
    return normalised_mse_radius(result.radius, condition["radius"], R0)


fit = fit_parameters(
    make_model,
    {
        "kappa_s": Parameter(1e-9, lower=0.0),
        "chi": Parameter(0.3, lower=0.0, upper=1.7),
    },
    conditions=conditions,
    loss_fn=loss_fn,
    optimizer=optax.adam(0.05),
    n_steps=100,
    save_spec=SAVE_SPEC,
    t_max=T_MAX,
)
print(fit.params)  # kappa_s about 3e-9, chi about 0.5
print(len(fit.result))  # 3: one SimulationResult per recording
```

When every condition has the same structure and the same array shapes, the
conditions run in parallel with `jax.vmap`, which is several times faster
than running them one after another. Otherwise, for example when recordings
have different numbers of frames, they run one after another, and the
compile time grows with the number of conditions. To keep many such
recordings in parallel, pad them to a common length and pass a mask in each
condition.

When the conditions run in parallel, each number in a condition, including a
Python int, reaches `make_model` and `loss_fn` as a traced array. Use it in
`jnp.where` or `jax.lax.cond`, not in an `if` or as a slice bound. A
condition that holds a Python bool or a string runs one after another, so
you can use these as configuration flags, for example
`if condition["with_shell"]:`.

To fit a value per recording, such as each bubble's equilibrium radius, use
an array `Parameter` and an index in each condition:
`Parameter(jnp.full(3, 2e-6), lower=0.0)` in `params0`, `"index": i` in each
condition, and `params["R0"][condition["index"]]` in `make_model`.

## Compare with camera frames or hydrophone signals

`loss_fn` receives the full `SimulationResult`, so you can compare any
differentiable quantity with your data.

To compare with camera frames, simulate on a fine time grid and interpolate
onto the frame times:

```{.python continuation}
frame_times = jnp.linspace(0.5e-6, 7.5e-6, 71)  # a 10 Mfps camera [s]
measured_frames = jnp.interp(
    frame_times, jnp.linspace(0.0, T_MAX, 256), measured_radius
)


def frame_loss(result):
    simulated = jnp.interp(frame_times, result.ts, result.radius)
    return normalised_mse_radius(simulated, measured_frames, R0)


fit = fit_parameters(
    make_model=lambda params: (make_eom(params["kappa_s"]), make_pulse(100e3)),
    params0={"kappa_s": Parameter(1e-9, lower=0.0)},
    loss_fn=frame_loss,
    optimizer=optax.adam(0.05),
    n_steps=100,
    save_spec=SaveSpec(1024),
    t_max=T_MAX,
    log_every=0,
)
```

To fit the radiated pressure at a hydrophone, compute the emission inside
`loss_fn`:

```{.python continuation}
from jbubble.acoustics import IncompressibleMonopole
from jbubble.metrics import normalised_mse_emission

emission = IncompressibleMonopole(rho_L=998.0)
r_hydrophone = 10e-3  # [m]
measured_p = emission(  # stand-in for a measured hydrophone signal [Pa]
    run_simulation(
        make_eom(3e-9), make_pulse(100e3), save_spec=SAVE_SPEC, t_max=T_MAX
    ),
    r_hydrophone,
)

fit = fit_parameters(
    make_model=lambda params: (make_eom(params["kappa_s"]), make_pulse(100e3)),
    params0={"kappa_s": Parameter(1e-9, lower=0.0)},
    loss_fn=lambda result: normalised_mse_emission(
        emission(result, r_hydrophone), measured_p, p_ref=1e3
    ),
    optimizer=optax.adam(0.05),
    n_steps=60,
    save_spec=SAVE_SPEC,
    t_max=T_MAX,
    log_every=0,
)
```

## Learn a constitutive law with a neural network

`params0` can be any Equinox module. The following
[`Property`][jbubble.bubble.property.Property] learns the surface tension
$\sigma(R)$ with a small network, bounded to $(0, \sigma_\text{max})$ by a
sigmoid. `fit_parameters` fits the network weights. `sigma_max` is a static
field, so it stays fixed:

```{.python continuation}
import equinox as eqx
import jax

from jbubble.bubble.property import Property


class BoundedNeuralSigma(Property):
    net: eqx.nn.MLP
    sigma_max: float = eqx.field(default=0.072, static=True)  # [N/m]

    def __call__(self, state):
        x = jnp.array([state.R / state.R0])
        return self.sigma_max * jax.nn.sigmoid(self.net(x)[0])


def make_neural_model(sigma, condition):
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.07),
        shell=LipidShell(sigma=sigma, kappa_s=3e-9),
        medium=NewtonianMedium(mu=1e-3),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )
    return eom, make_pulse(condition["pressure"])


sigma0 = BoundedNeuralSigma(net=eqx.nn.MLP(1, 1, 8, 2, key=jax.random.PRNGKey(0)))
fit = fit_parameters(
    make_neural_model,
    sigma0,
    conditions=conditions,
    loss_fn=loss_fn,
    optimizer=optax.adam(1e-2),
    n_steps=20,  # use thousands of steps in practice
    save_spec=SAVE_SPEC,
    t_max=T_MAX,
    log_every=0,
)
learned_sigma = fit.params  # a BoundedNeuralSigma with fitted weights
```

## Monitor and stop a fit

`step_callback(step, params, loss)` runs outside JIT for `params0`
(`step == 0`) and after every accepted step, with `params` in physical units.
Use it to record a trajectory, plot progress, save checkpoints, or stop early
by raising `StopIteration`:

```{.python continuation}
history = []


def callback(step, params, loss):
    history.append((step, float(params["kappa_s"]), loss))
    if loss < 1e-8:
        raise StopIteration  # converged


fit = fit_parameters(
    make_model=lambda params: (make_eom(params["kappa_s"]), make_pulse(100e3)),
    params0={"kappa_s": Parameter(1e-9, lower=0.0)},
    loss_fn=lambda result: normalised_mse_radius(result.radius, measured_radius, R0),
    optimizer=optax.adam(0.05),
    n_steps=500,
    save_spec=SAVE_SPEC,
    t_max=T_MAX,
    step_callback=callback,
    log_every=0,
)
print(fit.stopped_early, fit.message)
```

Any other exception from the callback, or a keyboard interrupt, ends the fit
without a `FitResult`. For a long fit, save the parameters from the callback
so that you can restart from them.

## Handle failed solves

A trial step can make the bubble collapse so violently that the solver
exceeds `SolverConfig.max_steps`, or make the loss or its gradient
non-finite. `fit_parameters` never raises from inside a solve. Instead, it
does the following:

- If a solve fails at `params0`, `fit_parameters` raises `RuntimeError`
  before the first step and names the failing condition. Start from a better
  initial guess, or raise `SolverConfig.max_steps`.
- If a step fails during the fit, `fit_parameters` halves the step and tries
  again, up to `max_backtracks` times (default 5). `fit.num_rejected` counts
  the halved steps.
- If every halved step fails, the fit warns, stops, and returns the last
  accepted parameters, with `fit.stopped_early` set to `True`.

Frequent rejections mean that the learning rate is too large or that a
parameter needs bounds. For example, `GompertzSurfaceTension` is defined only
for `chi * ((1 / R_buckle_ratio)**2 - 1) < sigma_rupture`, so bound `chi`
with `Parameter(..., upper=...)` when you fit it.

If the loss keeps falling toward a region where the solver fails, the fit
stops at the edge of that region. That's the expected result, not a bug:
`fit.params` holds the best parameters that the solver can simulate.

## Choose solver settings and an adjoint

By default `fit_parameters` uses [`SolverConfig()`][jbubble.solver.SolverConfig],
the same settings as [`run_simulation`][jbubble.simulation.run_simulation],
so the model that you fit is the model that you simulate.

| Setting | When to use it |
|---|---|
| `SolverConfig()` (default): `Dopri5`, `rtol=1e-6`, `atol=1e-10` on the scaled state | Most fits. |
| `SolverConfig(stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-8))` | Quick exploratory fits to noisy data. Each step is faster, but gradients can be a few percent off. |
| `SolverConfig.stiff()` | Stiff dynamics, for example a nanobubble or a very stiff shell, where `Dopri5` takes many tiny steps. |
| `adjoint=diffrax.RecursiveCheckpointAdjoint()` (default) | Gradient-based fits. Gives the exact gradient of the computed solution. |
| `adjoint=diffrax.ForwardMode()` | Forward-mode Jacobians with few parameters, for example Levenberg-Marquardt. |
| `adjoint=diffrax.BacksolveAdjoint()` | Only when memory runs out on very long integrations. Its gradients are approximate. |

## Use a least-squares solver

For a few physical parameters and data with a known noise level,
Levenberg-Marquardt from [optimistix](https://docs.kidger.site/optimistix/)
usually converges in tens of iterations instead of hundreds. Optimistix is
installed with diffrax. Levenberg-Marquardt needs residuals rather than a
scalar loss, and forward-mode derivatives through the solve:

```{.python continuation}
import diffrax
import optimistix as optx

from jbubble.fitting import unwrap

NOISE = 20e-9  # radius measurement noise, one standard deviation [m]


def residuals(params, args):
    p = unwrap(params)
    out = []
    for condition in conditions:
        eom, pulse = make_model(p, condition)
        result = run_simulation(
            eom, pulse, save_spec=SAVE_SPEC, t_max=T_MAX, adjoint=diffrax.ForwardMode()
        )
        out.append((result.radius - condition["radius"]) / NOISE)
    return jnp.concatenate(out)


params0 = {
    "kappa_s": Parameter(1e-9, lower=0.0),
    "chi": Parameter(0.3, lower=0.0, upper=1.7),
}
solution = optx.least_squares(
    residuals, optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8), params0, max_steps=100
)
print(unwrap(solution.value))  # kappa_s about 3e-9, chi about 0.5
```

## Estimate uncertainties

When the residuals are scaled by the measurement noise, the linearised
covariance of the fitted values follows from the Jacobian $J$ of the
residuals at the optimum:

$$
\operatorname{cov}(u) = (J^\top J)^{-1}, \qquad
\operatorname{cov}(x) = G \operatorname{cov}(u)\, G^\top, \qquad
G = \frac{\partial x}{\partial u},
$$

where $u$ holds the optimiser's coordinates and $x$ the physical values:

```{.python continuation}
from jax.flatten_util import ravel_pytree

u, unravel = ravel_pytree(solution.value)
J = jax.jacfwd(lambda v: residuals(unravel(v), None))(u)
G = jax.jacfwd(lambda v: ravel_pytree(unwrap(unravel(v)))[0])(u)
covariance = G @ jnp.linalg.inv(J.T @ J) @ G.T
stderr = jnp.sqrt(jnp.diag(covariance))  # same order as ravel_pytree: chi, kappa_s
print(dict(zip(["chi", "kappa_s"], stderr.tolist(), strict=True)))
```

These standard errors assume Gaussian noise with the stated `NOISE` and a
model that is close to linear near the optimum. Treat them with caution when
a value sits at a bound or when parameters are strongly correlated.

## Write your own training loop

`fit_parameters` runs a Python loop, so you can't call it under `jax.jit` or
`jax.vmap`, and each call compiles its functions again. For many fits, such as
multi-start fits or Monte Carlo studies, or for custom schedules and
regularisation terms, write the loop yourself. Compile the step once and
reuse it. [`unwrap`][jbubble.fitting.unwrap] turns a pytree of `Parameter`s
into physical values:

```{.python continuation}
params = {"kappa_s": Parameter(1e-9, lower=0.0)}
trainable, static = eqx.partition(params, eqx.is_inexact_array)
optimizer = optax.adam(0.05)
opt_state = optimizer.init(trainable)


@eqx.filter_jit
def train_step(trainable, opt_state):
    def loss(tr):
        p = unwrap(eqx.combine(tr, static))
        result = run_simulation(
            make_eom(p["kappa_s"]), make_pulse(100e3), save_spec=SAVE_SPEC, t_max=T_MAX
        )
        return normalised_mse_radius(result.radius, measured_radius, R0)

    value, grads = jax.value_and_grad(loss)(trainable)
    updates, opt_state = optimizer.update(grads, opt_state, trainable)
    return eqx.apply_updates(trainable, updates), opt_state, value


for _ in range(50):
    trainable, opt_state, value = train_step(trainable, opt_state)
print(unwrap(eqx.combine(trainable, static)))
```

A hand-written loop doesn't retry failed steps. Check `result.converged`, or
check that `value` is finite, before you apply an update.

Build each `Parameter` outside `jax.jit` and `jax.vmap`, because its
constructor checks the value and bounds on concrete numbers. To start several
fits from different guesses, build one `params` per guess and call the
compiled `train_step` on each. Give every guess the same explicit `scale`,
for example `Parameter(guess, lower=0.0, scale=1e-9)`. `scale` is a static
field, and by default it comes from the initial value, so guesses with
different default scales have different pytree structures. They don't match
the `static` that `train_step` closes over, and `eqx.combine` raises.

## Migrate from jbubble 0.1

| In 0.1 | In 0.2 |
|---|---|
| `params0=1e-9` was silently never fitted | Python floats are fitted: `1e-9` means `Parameter(1e-9)` |
| `params0=jnp.array(1e-9)` with `optax.adam(1e-10)` | `Parameter(1e-9, lower=0.0)` with `optax.adam(0.05)` |
| Hand-written sigmoid or logit transforms | `Parameter(value, lower=..., upper=...)` |
| One `(eom, pulse)` per fit | `conditions=[...]`, with `make_model(params, condition)` and `loss_fn(result, condition)` |
| A failed solve raised `RuntimeError` and ended the fit | The step is halved and retried |
| Default solver tolerances `rtol=1e-4`, `atol=1e-8` | `SolverConfig()`; pass `config=` for other settings |
| `save_spec` was required | `save_spec` defaults to `SaveSpec()` |
| `loss_history` had `n_steps` entries | `loss_history[0]` is the loss at `params0`, so it has one more entry |
| `step_callback` first ran after the first update | `step_callback` also runs at `step == 0` with `params0` |
| `fit.params` and callback `params` had the structure of `params0` | The same structure, with each `Parameter` replaced by its physical value |
| `fit.result` was one `SimulationResult` | With `conditions`, `fit.result` is a list with one `SimulationResult` per condition |
| `FitResult` was a mutable dataclass | `FitResult` is frozen; use `dataclasses.replace` to change a field |
