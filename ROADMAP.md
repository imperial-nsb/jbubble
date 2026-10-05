# Roadmap

This document lists the features and improvements planned for jbubble after
0.2.0. Plans change as research needs change, so none of these items has a
release date. For what each release contains, see the
[changelog](CHANGELOG.md).

## Physics models

- **Thermal gas, hysteretic shell, and gas diffusion models.** Exploratory
  work on these models lives on the `thermal-gas` branch. They join a release
  only after they pass physics regression tests against published references.
- **Confinement models.** jbubble 0.1 had experimental models for a bubble in
  a rigid tube and a bubble inside a spherical cavity, which 0.2.0 removes.
  Confinement models are planned to return with validated equations and
  tests.

## Simulation and sweeps

- **`GridSweep` on `jax.shard_map`.** `GridSweep` compiles one executable per
  device and runs chunks on worker threads. Sharding the grid with
  `jax.shard_map` compiles once for all devices, but implicit solvers such as
  `Kvaerno5` fail under sharding until two upstream pull requests are
  released: [lineax#246](https://github.com/patrick-kidger/lineax/pull/246)
  and [equinox#1261](https://github.com/patrick-kidger/equinox/pull/1261).
  The switch is planned after those releases, if it's faster than worker
  threads on GPUs. The `GridSweep` API stays the same.
- **Arbitrary sample times in `SaveSpec`.** `SaveSpec` records evenly spaced
  samples. Planned support for arbitrary sample times lets you compare a
  simulation with measurements at their own times, such as camera frames at
  an irregular frame rate.

## Fitting

- **Built-in Levenberg-Marquardt fitting.** The
  [fitting guide](https://imperial-nsb.github.io/jbubble/guide/fitting/) shows
  how to fit with a Levenberg-Marquardt solver yourself. A built-in option in
  `fit_parameters`, or a companion function, is planned.

## Documentation and infrastructure

- **Versioned documentation**, so the site keeps the docs of each release.
- **An interactive demo**, where you change a bubble's parameters and see its
  response without writing code.
- **Windows CI.** CI tests Linux and macOS. Testing on Windows too lets
  jbubble support it officially.

## How to contribute

To work on any of these items, first open an issue on
[GitHub](https://github.com/imperial-nsb/jbubble/issues) to discuss the
approach. For development guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
