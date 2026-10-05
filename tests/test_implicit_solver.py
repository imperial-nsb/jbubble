"""Regression tests for implicit (stiff) ODE solvers.

The default solver is explicit (Dopri5), so nothing else in the suite runs an
implicit solve. These tests guard against upstream regressions that only show
up there:

- jax>=0.8.2 with lineax<=0.0.8 raised ``TracerBoolConversionError`` inside
  Kvaerno solvers (lineax#187, fixed in lineax 0.1.0).
- diffrax 0.7.1 with optimistix 0.1.0 rejected every Kvaerno step.
- diffrax 0.7.2 with optimistix 0.1.0 takes about four times more Kvaerno5
  steps on the lipid_bubble preset than diffrax 0.7.1 with optimistix 0.0.11
  (3,003 vs 827). diffrax#754 fixes it on diffrax main, unreleased at the time
  of writing.
"""

from importlib.metadata import version

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jbubble import SaveSpec, SolverConfig, run_simulation, solve_eom
from jbubble.utils.presets import lipid_bubble


def _diffrax_version() -> tuple[int, ...]:
    return tuple(int(p) for p in version("diffrax").split(".")[:3] if p.isdigit())


def _peak_expansion(solver: diffrax.AbstractSolver):
    preset = lipid_bubble()
    config = SolverConfig(solver=solver)

    def peak(R0):
        eom = eqx.tree_at(lambda e: e.R0, preset.eom, R0)
        result = run_simulation(
            eom, preset.pulse, save_spec=SaveSpec(num_samples=256), config=config
        )
        return jnp.max(result.radius) / R0, result.converged

    return peak, jnp.asarray(preset.eom.R0)


@pytest.mark.parametrize(
    "solver", [diffrax.Kvaerno3(), diffrax.Kvaerno5()], ids=["Kvaerno3", "Kvaerno5"]
)
def test_implicit_solver_matches_explicit_under_jit_and_grad(solver):
    implicit, R0 = _peak_expansion(solver)
    explicit, _ = _peak_expansion(diffrax.Dopri5())

    value, converged = jax.jit(implicit)(R0)
    reference, _ = jax.jit(explicit)(R0)
    assert bool(converged)
    assert jnp.isclose(value, reference, rtol=1e-4)

    grad = jax.jit(jax.grad(lambda r: implicit(r)[0]))(R0)
    assert jnp.isfinite(grad)


# diffrax 0.7.2 requires optimistix>=0.1.0, and together they make Kvaerno5
# take about four times more steps than Dopri5 on this preset (3,003 vs 767).
# diffrax PR #754 fixes it on main, but no release contains it yet. The marker
# only covers diffrax 0.7.2 and older, so the CI jobs that test the newest
# releases run this as a normal test once a fixed diffrax ships; then raise the
# diffrax floor in pyproject.toml and delete the marker. The xfail isn't strict,
# so an optimistix release that fixes the step count shows up as an XPASS
# instead of a failure.
@pytest.mark.xfail(
    _diffrax_version() <= (0, 7, 2),
    reason="diffrax<=0.7.2 with optimistix 0.1.0 over-rejects implicit steps "
    "(fixed by diffrax#754; raise the diffrax floor once it is released)",
    strict=False,
)
def test_implicit_solver_step_count_is_comparable_to_explicit():
    preset = lipid_bubble()

    def steps(solver):
        sol = solve_eom(
            preset.eom,
            preset.pulse,
            save_spec=SaveSpec(num_samples=256),
            config=SolverConfig(solver=solver),
        )
        return int(sol.stats["num_steps"])

    assert steps(diffrax.Kvaerno5()) < 2 * steps(diffrax.Dopri5())
