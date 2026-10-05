"""Regression tests for implicit (stiff) ODE solvers.

The default solver is explicit (Dopri5), so nothing else in the suite runs an
implicit solve. These tests guard against upstream regressions that only show
up there:

- jax>=0.8.2 with lineax<=0.0.8 raised ``TracerBoolConversionError`` inside
  Kvaerno solvers (lineax#187, fixed in lineax 0.1.0).
- diffrax 0.7.1 with optimistix 0.1.0 rejected every Kvaerno step.
- diffrax 0.7.2 with optimistix 0.1.0 takes about six times more Kvaerno steps
  than before (fixed on diffrax main by diffrax#754, unreleased at the time of
  writing).
"""

from importlib.metadata import version

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jbubble import SaveSpec, SolverConfig, run_simulation, solve_eom
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import LipidShell, MarmottantSurfaceTension, NoShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
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
# take about six times more steps than Dopri5 on this preset (4,311 vs 684).
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


# ── SolverConfig.stiff on stiff problems ─────────────────────────────────────


def _lipid_nanobubble(R0, kappa_s=7.5e-9):
    sigma = MarmottantSurfaceTension(
        R_buckle_ratio=0.98058, chi=0.5, sigma_rupture=0.072
    )
    return KellerMiksis(
        gas=PolytropicGas(gamma=1.095),
        shell=LipidShell(sigma=sigma, kappa_s=kappa_s),
        medium=NewtonianMedium(mu=1e-3),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


def _viscous_submicron(mu=0.05):
    return KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=mu),
        R0=0.3e-6,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )


_STIFF_CASES = {
    "lipid-50nm": (lambda p: _lipid_nanobubble(50e-9, p), 7.5e-9, 5e6),
    "lipid-100nm": (lambda p: _lipid_nanobubble(100e-9, p), 7.5e-9, 5e6),
    "lipid-150nm": (lambda p: _lipid_nanobubble(150e-9, p), 7.5e-9, 5e6),
    "viscous-300nm": (_viscous_submicron, 0.05, 2e6),
}
_REFERENCE = SolverConfig(
    stepsize_controller=diffrax.PIDController(rtol=1e-11, atol=1e-13),
    max_steps=1_000_000,
)


def _stiff_solve(name, config, p=None):
    make_eom, p0, freq = _STIFF_CASES[name]
    pulse = ToneBurst(freq=freq, pressure=100e3, shape=Sine(), cycle_num=5)

    def solve(p):
        return solve_eom(make_eom(p), pulse, save_spec=SaveSpec(512), config=config)

    def loss(p):
        sol = solve(p)
        return jnp.mean((sol.ys.R / sol.ys.R0 - 1.0) ** 2)

    p = jnp.asarray(p0 if p is None else p)
    return jax.jit(solve)(p), jax.jit(jax.grad(loss))


@pytest.mark.parametrize("name", list(_STIFF_CASES))
def test_stiff_config_is_accurate_and_cheap_on_stiff_problems(name):
    """Kvaerno5 matches a tight reference with far fewer steps than Dopri5."""
    stiff, _ = _stiff_solve(name, SolverConfig.stiff())
    explicit, _ = _stiff_solve(name, SolverConfig())
    reference, _ = _stiff_solve(name, _REFERENCE)
    assert diffrax.is_successful(stiff.result)
    R0 = float(stiff.ys.R0[0])
    assert float(jnp.max(jnp.abs(stiff.ys.R - reference.ys.R))) < 2e-6 * R0
    assert 4 * int(stiff.stats["num_steps"]) < int(explicit.stats["num_steps"])


@pytest.mark.parametrize(
    ("R0", "kappa_s", "config"),
    [
        (20e-9, 2.5e-9, SolverConfig.stiff()),
        (100e-9, 7.5e-9, SolverConfig.stiff(rtol=1e-8, atol=1e-12)),
    ],
    ids=["20nm-default", "100nm-tight"],
)
def test_stiff_config_converges_on_tiny_bubbles_and_tight_tolerances(
    R0, kappa_s, config
):
    """The Newton floor keeps the implicit solve from stalling.

    Each case stalls in a different way without it: the 20 nm solve rejects
    most steps and reaches max_steps when the Newton atol is floored at only
    1e-8, and the tight 100 nm solve does so when the Newton tolerances are
    inherited from the controller (Kvaerno5's default VeryChord).
    """
    pulse = ToneBurst(freq=5e6, pressure=100e3, shape=Sine(), cycle_num=5)
    sol = jax.jit(
        lambda: solve_eom(
            _lipid_nanobubble(R0, kappa_s), pulse, save_spec=SaveSpec(64), config=config
        )
    )()
    assert diffrax.is_successful(sol.result)
    assert int(sol.stats["num_steps"]) < 5_000


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_STIFF_CASES))
def test_stiff_config_gradient_matches_tight_reference(name):
    """Through an implicit solve the gradient is accurate; through Dopri5 at
    its stability limit it can be wrong by orders of magnitude."""
    _, grad = _stiff_solve(name, SolverConfig.stiff())
    _, grad_reference = _stiff_solve(name, _REFERENCE)
    p0 = jnp.asarray(_STIFF_CASES[name][1])
    assert float(grad(p0)) == pytest.approx(float(grad_reference(p0)), rel=1e-4)
