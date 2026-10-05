# %% [markdown]
# # Equations of motion
#
# Solve one bubble with four equations of motion, from the incompressible
# Rayleigh-Plesset equation to the Gilmore equation for violent collapses.
# Then raise the drive pressure to see where stable oscillation gives way to
# inertial cavitation.
#
# All four equations share the same gas, shell, and medium models; they
# differ in how they treat the liquid around the bubble.

# %%
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jbubble import SaveSpec, run_simulation
from jbubble.bubble.eom import (
    Gilmore,
    KellerMiksis,
    ModifiedRayleighPlesset,
    RayleighPlesset,
)
from jbubble.bubble.gas import VanDerWaalsGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.pulse import HannEnvelope, ToneBurst
from jbubble.pulse.shapes import Sine

plt.style.use("jbubble.style.light")
DRIVE = "#8c959f"  # neutral grey for the acoustic drive in the light theme
QUICK = os.environ.get("JBUBBLE_QUICK") == "1"  # a smaller sweep for CI

# %% [markdown]
# ## Build four equations of motion
#
# - `RayleighPlesset` treats the liquid as incompressible. It has no way to
#   lose energy to sound, so it overestimates violent growth and collapse.
# - `ModifiedRayleighPlesset` adds a first-order correction for the sound
#   that the gas pressure radiates, as in Marmottant et al. (2005).
# - `KellerMiksis` keeps the liquid's compressibility to first order in the
#   Mach number of the bubble wall. The presets use it.
# - `Gilmore` describes the liquid with the Tait equation of state, which
#   stays accurate when the wall moves at a sizeable fraction of the speed
#   of sound.
#
# The bubble is a 2 µm air bubble in water. A van der Waals gas keeps the
# collapse physical: the gas can't be compressed below the volume of its
# molecules, a hard core of radius $R_0 / 8.86$. Gilmore's Tait constants
# set the speed of sound in water, so the other equations use the same value.

# %%
parts = dict(
    gas=VanDerWaalsGas(gamma=1.4, h_frac=1 / 8.86),
    shell=NoShell(sigma=0.072),
    medium=NewtonianMedium(mu=1e-3),
    R0=2e-6,
    P_amb=101325.0,
    rho_L=998.0,
)
gilmore = Gilmore(**parts)  # default Tait constants for water
c_water = float(
    jnp.sqrt(gilmore.n_tait * (gilmore.P_amb + gilmore.B_tait) / gilmore.rho_L)
)
print(f"Speed of sound from the Tait constants: {c_water:.0f} m/s")

models = {
    "Keller-Miksis": KellerMiksis(**parts, c_L=c_water),
    "Rayleigh-Plesset": RayleighPlesset(**parts),
    "modified Rayleigh-Plesset": ModifiedRayleighPlesset(**parts, c_L=c_water),
    "Gilmore": gilmore,
}

# %% [markdown]
# ## Compare them at 400 kPa
#
# Drive the bubble with three cycles at 1 MHz and 400 kPa under a Hann
# window. At this pressure the bubble grows to several times its size and
# then collapses, which is where the equations disagree most. Record 4000
# samples, because each collapse lasts only a few nanoseconds.

# %%
pulse = ToneBurst(
    freq=1e6, pressure=400e3, shape=Sine(), cycle_num=3, envelope=HannEnvelope()
)
runs = {}
for name, model in models.items():
    start = time.perf_counter()
    runs[name] = run_simulation(
        model, pulse, t_max=6e-6, save_spec=SaveSpec(num_samples=4000)
    )
    elapsed = time.perf_counter() - start
    R = runs[name].radius / model.R0
    print(
        f"{name:26s} R/R0 from {R.min():.3f} to {R.max():.2f}, "
        f"converged: {bool(runs[name].converged)}, {elapsed:.1f} s with compilation"
    )

# %%
fig, (ax_p, ax_r) = plt.subplots(
    2, 1, figsize=(7.5, 4.6), sharex=True, height_ratios=[1, 2.4]
)
first = runs["Keller-Miksis"]
ax_p.plot(first.ts * 1e6, first.driving_pressure / 1e3, color=DRIVE)
ax_p.set_ylabel("drive (kPa)")
ax_p.set_title("One bubble, four equations of motion: 1 MHz, 400 kPa")
ax_r.axhline(1.0, color=DRIVE, lw=0.8, ls="--")
# Keller-Miksis is wide and underneath; Gilmore is dashed on top of it.
styles = [dict(lw=3.0), dict(lw=1.4), dict(lw=1.4), dict(lw=1.4, ls="--")]
for colour, style, (name, run) in zip(
    ["C0", "C1", "C2", "C3"], styles, runs.items(), strict=True
):
    ax_r.plot(run.ts * 1e6, run.radius / parts["R0"], color=colour, label=name, **style)
ax_r.set_xlabel("time (µs)")
ax_r.set_ylabel("$R / R_0$")
ax_r.legend(loc="upper left")
plt.show()

# %% [markdown]
# Rayleigh-Plesset lets the bubble grow furthest, because an incompressible
# liquid can't carry energy away as sound. The three equations that radiate
# sound agree closely at this pressure; they differ mainly in how deep each
# collapse goes. As collapses grow more violent and the wall approaches the
# speed of sound, Gilmore becomes the safer choice.
#
# ## Find the onset of inertial cavitation
#
# At low pressures the bubble oscillates gently around its equilibrium
# radius for as long as the drive lasts: stable cavitation. Above a
# threshold it grows to several times its size and collapses violently under
# the inertia of the liquid that rushes in: inertial cavitation. A common
# rule of thumb calls a collapse inertial when the bubble first grows to at
# least twice its equilibrium radius.
#
# `jax.vmap` runs the Keller-Miksis model at every pressure in one compiled
# call. A sweep function returns the peak radius together with the solver's
# `converged` flag, so you can discard any run that failed.

# %%
keller_miksis = models["Keller-Miksis"]
pressures = jnp.linspace(10e3, 400e3, 16 if QUICK else 60)


def peak_radius(pressure):
    drive = ToneBurst(
        freq=1e6, pressure=pressure, shape=Sine(), cycle_num=3, envelope=HannEnvelope()
    )
    run = run_simulation(
        keller_miksis, drive, t_max=6e-6, save_spec=SaveSpec(num_samples=2048)
    )
    return run.radius.max() / keller_miksis.R0, run.converged


start = time.perf_counter()
peaks, converged = jax.jit(jax.vmap(peak_radius))(pressures)
peaks.block_until_ready()
print(
    f"{pressures.size} pressures in {time.perf_counter() - start:.1f} s, all converged: {bool(converged.all())}"
)

inertial = peaks >= 2.0
threshold = (
    float(pressures[jnp.argmax(inertial)]) if bool(inertial.any()) else float("nan")
)
print(f"Peak R/R0 first reaches 2 at about {threshold / 1e3:.0f} kPa")

# %% [markdown]
# Plot two runs on either side of the threshold beside the peak radius at
# every pressure.

# %%
examples = {"stable, 60 kPa": 60e3, "inertial, 300 kPa": 300e3}
fig = plt.figure(figsize=(9.0, 4.2))
grid = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
ax_peak = fig.add_subplot(grid[:, 1])
for row, (label, pressure) in enumerate(examples.items()):
    drive = ToneBurst(
        freq=1e6, pressure=pressure, shape=Sine(), cycle_num=3, envelope=HannEnvelope()
    )
    run = run_simulation(
        keller_miksis, drive, t_max=6e-6, save_spec=SaveSpec(num_samples=4000)
    )
    ax = fig.add_subplot(grid[row, 0])
    ax.axhline(1.0, color=DRIVE, lw=0.8, ls="--")
    ax.plot(run.ts * 1e6, run.radius / keller_miksis.R0, color="C0")
    ax.set_ylabel("$R / R_0$")
    ax.set_title(label[0].upper() + label[1:])
    peak = float(run.radius.max() / keller_miksis.R0)
    ax_peak.plot(pressure / 1e3, peak, "o", color="C0", mfc="none", ms=9)
    offset, align = ((8, -12), "left") if row == 0 else ((-10, 6), "right")
    ax_peak.annotate(
        label.split(",")[0],
        (pressure / 1e3, peak),
        textcoords="offset points",
        xytext=offset,
        ha=align,
    )
ax.set_xlabel("time (µs)")

ax_peak.plot(pressures / 1e3, peaks, color="C0", marker="o", ms=3)
ax_peak.axhline(2.0, color="C1", lw=1.0, ls="--")
ax_peak.text(12, 2.08, "$R_\\mathrm{max} = 2 R_0$", color="C1", fontsize=9)
ax_peak.set_xlabel("drive pressure (kPa)")
ax_peak.set_ylabel("peak $R / R_0$")
ax_peak.set_title("Keller-Miksis, 1 MHz, three cycles")
plt.show()
