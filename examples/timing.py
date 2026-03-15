"""jbubble timing — demonstrates JIT and vmap acceleration.

Shows wall-clock speedups for three progressively faster approaches:

  1. Eager (no JIT):   single simulation, no compilation
  2. JIT:              single simulation compiled once, fast on repeat calls
  3. JIT + vmap:       N simulations batched and compiled together

Run with:
    python examples/timing.py
"""

import time

import jax
import jax.numpy as jnp

from jbubble import run_simulation
from jbubble.bubble.eom import KellerMiksis
from jbubble.bubble.gas import PolytropicGas, VanDerWaalsGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import LipidShell, GompertzSurfaceTension
from jbubble.pulse import ToneBurst, HannEnvelope, Summed
from jbubble.pulse.base import TukeyEnvelope
from jbubble.pulse.chirp import ChirpPulse
from jbubble.pulse.shapes import Sine
from jbubble.solver import SaveSpec

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

SAVE_SPEC = SaveSpec(num_samples=512)
WINDOW_S = 20e-6
N_REPEAT = 5          # how many times to re-run JIT-compiled calls for timing
N_BATCH = 32          # number of simulations in the vmap batch

R0_VALUES = jnp.linspace(1e-6, 5e-6, N_BATCH)   # sweep bubble radii 1–5 µm

sigma = GompertzSurfaceTension(
    R_buckle_ratio=0.99,
    chi=0.38,
    sigma_break=72e-3,
)
shell  = LipidShell(sigma=sigma, kappa_s=2.4e-9)
gas = VanDerWaalsGas(gamma=1.07, h_frac=1 / 5.61)
medium = NewtonianMedium(mu=0.00089)

pulseA = ToneBurst(
    freq=1e6,
    pressure=200e3,
    shape=Sine(),
    cycle_num=15.0,
    envelope=HannEnvelope(),
)

pulseB = ChirpPulse(
    freq_start=0.2e6,
    freq_end=2e6,
    pressure=200e3,
    sweep_duration=14e-6,
    # envelope=HannEnvelope(),
)

# pulse = Summed((pulseA, pulseB), envelope=HannEnvelope(), initial_time=0e-6)
# pulse = pulseA + 10 * pulseB
pulse = 10 * (pulseA + 3.4 * pulseB).windowed(HannEnvelope())
print(type(pulse))

def make_eom(R0: float) -> KellerMiksis:
    return KellerMiksis(
        gas=gas,
        shell=shell,
        medium=medium,
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1481.0,
    )


def _block(result):
    """Block until JAX has finished computing, then return result."""
    jax.block_until_ready(result.radius)
    return result


def _time(fn, repeats: int = 1) -> tuple[float, object]:
    """Run fn() `repeats` times and return (mean wall time, last result)."""
    result = None
    t_total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        _block(result)
        t_total += time.perf_counter() - t0
    return t_total / repeats, result


# ---------------------------------------------------------------------------
# 1. Eager — no JIT, single simulation
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Eager (no JIT) — single simulation")
print("=" * 60)

eom_single = make_eom(2e-6)

t_eager, result_eager = _time(
    lambda: run_simulation(eom_single, pulse, save_spec=SAVE_SPEC, window_s=WINDOW_S),
    repeats=1,
)
print(f"   R0        : {eom_single.R0 * 1e6:.1f} µm")
print(f"   Converged : {bool(result_eager.converged)}")
print(f"   Wall time : {t_eager * 1e3:.1f} ms")

# ---------------------------------------------------------------------------
# 2. JIT — single simulation, compiled
# ---------------------------------------------------------------------------

# print()
# print("=" * 60)
# print("2. JIT — single simulation, compiled")
# print("=" * 60)

# jit_run = jax.jit(run_simulation)

# # Warm-up (compilation happens here)
# t_warmup, _ = _time(
#     lambda: jit_run(eom_single, pulse, save_spec=SAVE_SPEC, window_s=WINDOW_S),
#     repeats=1,
# )
# print(f"   Warmup (compile + run) : {t_warmup * 1e3:.1f} ms")

# # Subsequent calls — only execution, no recompilation
# t_jit, result_jit = _time(
#     lambda: jit_run(eom_single, pulse, save_spec=SAVE_SPEC, window_s=WINDOW_S),
#     repeats=N_REPEAT,
# )
# print(f"   Mean over {N_REPEAT} calls       : {t_jit * 1e3:.2f} ms")
# print(f"   Speedup vs eager       : {t_eager / t_jit:.1f}×")

# # ---------------------------------------------------------------------------
# # 3. JIT + vmap — N simulations batched together
# # ---------------------------------------------------------------------------

# print()
# print("=" * 60)
# print(f"3. JIT + vmap — {N_BATCH} simulations in parallel")
# print("=" * 60)


# def kernel(R0):
#     """Simulate a single bubble at equilibrium radius R0."""
#     eom = make_eom(R0)
#     return run_simulation(eom, pulse, save_spec=SAVE_SPEC, window_s=WINDOW_S)


# jit_vmap_run = jax.jit(jax.vmap(kernel))

# # Warm-up
# t_vmap_warmup, _ = _time(lambda: jit_vmap_run(R0_VALUES), repeats=1)
# print(f"   Warmup (compile + run) : {t_vmap_warmup * 1e3:.1f} ms")

# # Steady-state timing
# t_vmap, result_batch = _time(lambda: jit_vmap_run(R0_VALUES), repeats=N_REPEAT)
# print(f"   Mean over {N_REPEAT} calls       : {t_vmap * 1e3:.2f} ms")
# print(f"   Per-simulation          : {t_vmap / N_BATCH * 1e3:.3f} ms")
# print(f"   Speedup vs eager×{N_BATCH}     : {t_eager * N_BATCH / t_vmap:.1f}×")
# print(f"   Speedup vs jit×{N_BATCH}       : {t_jit * N_BATCH / t_vmap:.1f}×")

# # Sanity-check results
# converged = result_batch.converged
# r_max = result_batch.radius.max(axis=-1)

# print()
# print(f"   R0 range  : {float(R0_VALUES[0]) * 1e6:.1f} – {float(R0_VALUES[-1]) * 1e6:.1f} µm")
# print(f"   Converged : {int(converged.sum())}/{N_BATCH}")
# print(f"   R_max range: {float(r_max.min()) * 1e6:.2f} – {float(r_max.max()) * 1e6:.2f} µm")

# # ---------------------------------------------------------------------------
# # Summary
# # ---------------------------------------------------------------------------

# print()
# print("=" * 60)
# print("Summary")
# print("=" * 60)
# print(f"  Eager (1 sim)               : {t_eager * 1e3:8.2f} ms")
# print(f"  JIT   (1 sim, compiled)     : {t_jit * 1e3:8.3f} ms  ({t_eager / t_jit:.1f}× faster)")
# print(f"  JIT+vmap ({N_BATCH} sims, compiled) : {t_vmap * 1e3:8.2f} ms  "
#       f"({t_eager * N_BATCH / t_vmap:.1f}× faster than {N_BATCH}× eager)")
