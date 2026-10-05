# Changelog

This file lists the notable changes in each jbubble release.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and jbubble uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, a minor release can contain breaking changes.

## [0.2.0] - Unreleased

jbubble 0.2.0 is a beta release. It corrects several physics models against
published references, keeps gradients finite through the adaptive solver, runs
parameter sweeps on every CPU core, redesigns parameter fitting, and adds a
documentation site with an example gallery. It contains breaking changes:
[Upgrading from 0.1](#upgrading-from-01) lists each one and what to change.

### Upgrading from 0.1

Installation:

- jbubble needs Python 3.12 or later.
- h5py is no longer a core dependency. To use `jbubble.utils.io`, install the
  `io` extra: `pip install "jbubble[io]"`.
- The `dev` and `docs` extras are gone. In a source checkout, use the
  dependency groups instead: `uv sync` or `uv sync --group docs`, or
  `pip install -e ".[io,examples]" --group dev` with pip 25.1 or later.
- In a source checkout, delete the `jbubble/_version.py` file and the
  `jbubble.egg-info/` directory that setuptools-scm left behind. Otherwise
  `jbubble.__version__` can report the old version.

Models:

- `LeightonTube`, `SphericalConfinement`, and `ConfinedBubbleState` are
  removed, with no replacement in this release.
- `ThickShell`, `KelvinVoigtMedium`, `PowerLawMedium`, and
  `GompertzSurfaceTension` now follow their published equations, so they give
  different results. Rerun simulations and refit parameters that use them.
- `GompertzSurfaceTension` raises `ValueError` at construction unless
  0 < σ(R0) < `sigma_rupture`, `chi` > 0, and 0 < `R_buckle_ratio` < 1.
- `chi` and `sigma_rupture` on `MarmottantSurfaceTension` and
  `GompertzSurfaceTension` are `Property` fields. The constructors still
  accept floats. To replace one with `eqx.tree_at`, pass a `Property` or
  target its `.val`, as in `lambda s: s.chi.val`.
- The presets have new, cited defaults. `lipid_bubble` uses
  `SmoothMarmottantSurfaceTension` instead of `GompertzSurfaceTension`, and
  SonoVue parameters from Gümmer et al. (2021). `thick_shell_bubble` uses the
  polymer-shell parameters of Hoff et al. (2000). To keep a specific value,
  pass it as a keyword argument.
- `Gilmore` defaults to the Tait constants `n_tait=7.15` and
  `B_tait=3.046e8`. To keep the 0.1 values, pass `n_tait=7.0` and
  `B_tait=304.9e6`.
- `QuasiAcoustic` returns the monopole series at the solver's sample times.
  Plot it against `emission.observer_time(result, r)`, which is
  `result.ts + r / c_L`, instead of against `result.ts`.

Pulses:

- `Summed` has no window of its own, so a sum equals the sum of its children.
  To keep the 0.1 behaviour, call `.windowed(SoftRectangularEnvelope())` on
  the sum.
- `Scaled` and `Offset` raise `ValueError` for a non-default `initial_time` or
  `envelope`, which 0.1 ignored. Set `initial_time` on the wrapped pulse, or
  call `.windowed(envelope)`.
- `SampledPulse` raises `ValueError` when `initial_time` is neither 0.0 nor
  `ts[0]`. To delay a sampled waveform, shift `ts`.
- `NeuralPulse` stores `pulse_duration` and `pressure_scale` as static Python
  floats, so you can't trace, vmap over, or `eqx.tree_at` them. A
  `NeuralPulse` with a non-zero `initial_time` now delays the waveform, so the
  same weights give a different signal than in 0.1.

Solver and simulation:

- The default solver is `Dopri5` instead of `Kvaerno5`, with `rtol=1e-6`,
  `atol=1e-10`, and `max_steps=100_000`. In 0.1 they were `rtol=1e-3`,
  `atol=1e-6`, and `max_steps=10_000`. For stiff problems, such as lipid
  nanobubbles, use `SolverConfig.stiff()`.
- The step-size tolerances apply to a dimensionless state: the radius in units
  of R0 and the velocity in units of sqrt(P_amb / rho_L). If you set `atol`
  in metres, divide it by R0.
- `run_simulation` warns about a failed solve only when you call it eagerly.
  Under `jax.jit` or `jax.vmap`, check `result.converged`.

Sweeps:

- `GridSweep.run()` returns NumPy arrays, and `GridSweep.batches()` yields
  NumPy parameters and outputs, instead of `jax.Array`.
- `GridSweep` runs `fn` on worker threads. `batch_size` rounds up to a
  multiple of the worker count, and results can depend on the chunk size. To
  get the same chunks on every machine, set `workers` and `batch_size`.
  Worker threads don't inherit `with jax.enable_x64(...)` or
  `with jax.debug_nans(...)`; set these with `jax.config.update`.

Fitting:

- `FitResult.loss_history` has one entry per accepted step plus one:
  `loss_history[0]` is the loss at `params0`. `FitResult` is frozen.
- `step_callback` also runs at step 0.
- `fit_parameters` fits Python floats in `params0`, which 0.1 held fixed.
- When a solve fails or the loss isn't finite, `fit_parameters` halves the
  step and, after `max_backtracks` halvings, stops with a warning instead of
  raising. Check `FitResult.stopped_early` and `FitResult.message`.

### Added

- `SmoothMarmottantSurfaceTension(R_buckle_ratio, chi, sigma_rupture,
  smoothing=0.01)`, the Marmottant law with smoothed corners, for
  gradient-based fitting. It stays within `smoothing * ln 2 * sigma_rupture`
  of the piecewise law. Near buckling or rupture at R0, such as
  `R_buckle_ratio` close to 1, it biases results by up to a few percent; use
  `smoothing=0.005` there.
- `SolverConfig.stiff()`, an implicit `Kvaerno5` configuration with a `Chord`
  root finder, for lipid nanobubbles, sub-micron bubbles in viscous liquids,
  and other stiff problems.
- The `adjoint=` argument to `run_simulation`, for example
  `diffrax.ForwardMode()` for `jax.jacfwd`.
- `save_spec` is optional in `run_simulation` and `fit_parameters`, and
  defaults to `SaveSpec()`.
- The `R=` and `R_dot=` keyword arguments to
  `EquationOfMotion.initial_state`, to start away from equilibrium.
- The `EquationOfMotion.is_admissible`, `GasModel.is_admissible`, and
  `EquationOfMotion.state_scale` hooks. `VanDerWaalsGas` excludes its hard
  core.
- `EmissionModel.observer_time(result, r)`.
- `NoEnvelope`, a window that is 1 at all times.
- `Pulse.t_start`, `Pulse.t_stop`, and `Pulse.window_edges`, the start, end,
  and window edges of a pulse in seconds, delegated through `Scaled`,
  `Offset`, and `Summed`.
- The `devices` and `workers` arguments to `GridSweep`, and its `per_worker`,
  `workers`, `devices`, and `num_batches` attributes.
- `jbubble.fitting.Parameter`, which fits a value on a scaled coordinate with
  optional bounds or a fixed value, so one learning rate suits every
  parameter.
- `jbubble.fitting.unwrap`, which replaces each `Parameter` in a pytree with
  its physical value, for your own training loops.
- The `conditions=` argument to `fit_parameters`, which fits shared
  parameters to several recordings, in parallel under `jax.vmap` when the
  recordings have the same shapes, and the `max_backtracks=` argument.
- `FitResult.num_rejected`, `FitResult.stopped_early`, and
  `FitResult.message`.
- `jbubble.__version__`.
- The `jbubble.style.light` and `jbubble.style.dark` Matplotlib styles. Load
  one with `plt.style.use("jbubble.style.light")`.
- A documentation site at <https://imperial-nsb.github.io/jbubble/>, with
  guides, including a new guide to fitting model parameters to data, and an
  API reference.
- An example gallery with 11 new examples, from a first simulation to
  learning a shell law with a neural network. Each example is a jupytext
  percent-format script in `examples/`, and runs as a notebook in Google
  Colab.
- Animated README figures, generated by `scripts/make_readme_assets.py`.
- `scripts/bench_gridsweep.py`, which compares `GridSweep` worker counts and
  checks that their outputs are identical.
- Physics regression tests against independent references: the linearised
  resonance and damping of each equation of motion, and a hand-coded
  Keller-Miksis trajectory.
- This changelog, `CITATION.cff`, and a roadmap.

### Changed

- `GridSweep` runs grid chunks in parallel on worker threads, by default one
  per CPU core available to the process, up to 31 per CPU device. You no
  longer need `JAX_NUM_CPU_DEVICES` or `XLA_FLAGS` to use several cores. On a
  GPU or TPU, it uses every local device.
- `GridSweep` traces `fn` once and pads a ragged last batch with copies of
  the last grid point, so the last batch doesn't compile again. Chunks stream
  across batch boundaries, so a slow chunk holds up only its own worker. If
  you stop iterating over `batches()` early, it cancels queued chunks.
- The `GridSweep` progress bar uses `tqdm.auto`, which renders in notebooks,
  and counts grid points.
- `GompertzSurfaceTension` implements Eq. 15 of Gümmer, Schenke & Denner
  (2021) exactly. In 0.1 its rate constant lacked a factor of e, so the law
  rose at 0.37 times the published rate. It validates concrete values at
  construction and skips traced values, such as inside `jax.jit` or
  `jax.grad`.
- `chi` and `sigma_rupture` are `Property` fields on
  `MarmottantSurfaceTension`, `SmoothMarmottantSurfaceTension`, and
  `GompertzSurfaceTension`, so they can depend on the bubble state.
- `lipid_bubble` uses `SmoothMarmottantSurfaceTension` and the SonoVue
  parameters of Gümmer et al. (2021): `chi` 0.5 N/m, `kappa_s`
  7.5e-9 N s/m, `R_buckle_ratio` 0.98058, and `gamma` 1.095 for SF6. It has a
  new `smoothing` keyword. At 100 kPa and 1 MHz, the peak expansion of a 2 µm
  bubble falls from 42.9% to 22.7% of R0.
- `thick_shell_bubble` uses the polymer-shell parameters of Hoff et al.
  (2000): `G_s` 11.7 MPa, `mu_s` 0.45 Pa s, and a shell thickness `d_s` of 5%
  of R0 by default.
- Every preset docstring cites the source of each default, and the presets
  share one liquid: water at 20 °C.
- `Gilmore` defaults to the Tait constants n = 7.15 and B = 3.046e8 Pa
  (Gümmer et al. 2021), and documents `c_inf`.
- `PowerLawMedium` uses a smooth shear-rate floor, with a default `eps` of
  1e2 s^-1.
- `QuasiAcoustic` returns the monopole series unchanged, to plot against the
  observer time `ts + r / c_L`, instead of interpolating the radius,
  velocity, and acceleration at retarded times.
- The default solver is `Dopri5`, with `rtol=1e-6` and `atol=1e-10` on a
  dimensionless state, and `max_steps` rises from 10_000 to 100_000.
- `solve_eom` makes an adaptive step-size controller step to every pulse
  window edge.
- Pulse arithmetic accepts JAX scalars, including traced values, so you can
  differentiate, jit, or vmap over a factor or an offset. `pulse + jnp_scalar`
  builds an `Offset`.
- Pulse arithmetic raises `TypeError` for an operand that isn't a pulse or a
  number, and `ValueError` for a non-scalar array. `numpy_array * pulse` no
  longer builds an object array of pulses.
- Adding pulses keeps constant offsets outside the sum, so a window on the sum
  never gates a constant.
- `fit_parameters` fits Python floats in `params0`, and warns about Python
  floats inside Equinox modules, which stay fixed.
- `fit_parameters` halves a step whose solve fails or whose loss or gradient
  isn't finite, and stops with a warning instead of raising mid-fit.
- `fit_parameters` defaults to `SolverConfig()` and `SaveSpec()`, like
  `run_simulation`.
- Docstrings use Markdown numpy style with LaTeX equations, and each module
  lists its public names in `__all__`.
- Packaging uses hatchling and hatch-vcs instead of setuptools and
  setuptools-scm, and the repository has a `uv.lock` lockfile. Development
  uses [uv](https://docs.astral.sh/uv/), with Python 3.13 by default.
- Dependencies: the `jax==0.8.1` and `jaxlib==0.8.1` pins become `jax>=0.8.1`,
  `diffrax>=0.7.2`, `equinox>=0.13.4`, a `lineax>=0.1.0` floor, and a
  declared `optimistix>=0.1.0`. Matplotlib moves to the new `examples` extra.
- CI tests Python 3.12 with the oldest supported dependencies and with the
  lockfile, and Python 3.13 and 3.14 with the newest releases on Linux and
  macOS, every week as well as on each change. It also smoke-tests the built
  wheel. The docs workflow executes the examples and runs the code in the
  README and the guides.
- A release publishes its documentation only after PyPI has the release.

### Deprecated

- Nothing. jbubble is in beta, so this release changes and removes APIs
  without a deprecation period. [Upgrading from 0.1](#upgrading-from-01)
  lists each breaking change.

### Removed

- The confinement models `LeightonTube` and `SphericalConfinement`, and
  `ConfinedBubbleState`.
- Support for Python 3.11.
- The `chex` dependency, and h5py as a core dependency.
- The `dev` and `docs` extras, replaced by dependency groups.
- The 0.1 example scripts, replaced by the example gallery.

### Fixed

- `GompertzSurfaceTension` no longer deadlocks batched simulations. In 0.1,
  its construction check ran JAX operations inside a `jax.debug.callback`,
  which could intermittently hang `jax.vmap` runs and `GridSweep` sweeps. Every
  sweep over the 0.1 `lipid_bubble` preset was exposed. jbubble now runs no
  host callbacks inside traced code.
- `run_simulation` no longer adds a host callback under `jax.jit` and
  `jax.vmap`.
- `ThickShell` stresses were a factor of three too small. It now uses the
  finite-deformation Church model of Qin & Ferrara (2010), whose thin-shell
  limit is Hoff et al. (2000). For the 0.1 thick-shell preset values, the
  resonance moves from 2.33 MHz to 3.05 MHz.
- The `KelvinVoigtMedium` elastic term is (4G/3)(1 - (R0/R)^3), as in Yang &
  Church (2005).
- `PowerLawMedium` uses the rheometric shear rate 2 sqrt(3) |R_dot| / R.
- Gradients are no longer NaN when an adaptive solver tries and rejects a step
  that leaves the physical domain. The forward solution with explicit solvers
  is unchanged, bit for bit.
- The adaptive solver no longer steps over a pulse that starts late, which
  left the bubble at rest while reporting `converged = True`.
- A user-supplied `state0` with an unset or integer zero `R0` or `P_gas0` is
  filled from the equation of motion, instead of giving a NaN trajectory.
- A `Summed` pulse can be traced, so you can simulate a sum of pulses. In 0.1
  every sum failed under tracing.
- Calling a sum of pulses with an array of times returns the sum at each time,
  not one value repeated.
- A delayed pulse inside a sum is no longer zeroed when it's scaled, negated,
  or offset, as in `p1 - p2` or `p1 + 0.5 * p2`.
- `windowed()` on a `Scaled` or `Offset` pulse windows the wrapped pulse
  instead of doing nothing, and adding a pulse to a windowed sum keeps the
  sum's window.
- `SampledPulse` gates its signal from its first to its last sample time, so
  samples that start after 0 are no longer cut off early.
- `NeuralPulse` normalises time as `(t - initial_time) / pulse_duration`, so
  `initial_time` delays the waveform instead of changing it.
- `export_hdf5` accepts NumPy and JAX scalar metadata, and writes nothing if
  the metadata can't be serialised.
- `fit_parameters` no longer compiles twice for a weakly typed initial array.

## [0.1.1] - 2026-04-09

### Fixed

- Type annotations for `solve_eom` and `Pulse.duration`.
- README badges and links, and the package classifiers.

## [0.1.0] - 2026-03-18

First public release.

[0.2.0]: https://github.com/imperial-nsb/jbubble/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/imperial-nsb/jbubble/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/imperial-nsb/jbubble/releases/tag/v0.1.0
