"""Render the README animations and images into `docs/assets/readme/`.

Each asset has one function that simulates once and then renders a light and a
dark variant with the `jbubble.style` themes:

- `hero-bubble`: `hero-bubble-{light,dark}.gif`, a bubble drawn to scale
  beside its radius trace and the drive.
- `gradient-climb`: `gradient-climb-{light,dark}.gif`, `jax.grad` and optax
  climbing a `GridSweep` response map to its peak.
- `neural-sigma`: `neural-sigma-{light,dark}.gif`, a neural network learning
  the shell law sigma(R) from radius traces, against the Marmottant law.
- `emission-spectrum`: `emission-spectrum-{light,dark}.png`, the spectrum of
  the radiated pressure as the drive passes the broadband threshold.

To regenerate every asset, install the `assets` dependency group and run this
command from the repository root. It takes about three minutes on a laptop CPU:

    uv run --group assets python scripts/make_readme_assets.py

To render some of the assets, pass their names to `--only`. To check that
the script runs, pass `--quick`: it shrinks every computation and writes to a
temporary directory unless you also pass `--out`.

The script writes `manifest.json` next to the assets. For each file, it
records the size, the frame count, the duration, a hash of the inputs (this
script, the `jbubble` package, and `uv.lock`), the jbubble version, and the
git commit. It fails if an asset exceeds the size, width, or duration budget.

Every GIF opens on an informative frame, because GitHub shows only the first
frame to readers who turn off autoplay. The simulations use fixed parameters
and seeds, so a rerun reproduces the same frames.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import equinox as eqx
import imageio_ffmpeg
import jax
import jax.numpy as jnp
import jbubble
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
from jbubble import SaveSpec, fit_parameters, run_simulation
from jbubble.bubble.eom import KellerMiksis, RayleighPlesset
from jbubble.bubble.gas import PolytropicGas, VanDerWaalsGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.property import NeuralProperty
from jbubble.bubble.shell import LipidShell, MarmottantSurfaceTension, NoShell
from jbubble.bubble.state import BubbleState
from jbubble.metrics import normalised_mse_radius
from jbubble.pulse import HannEnvelope, ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.utils import GridSweep
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "assets" / "readme"

# Every asset is 8 in x 200 dpi = 1600 px wide. The README shows it at about
# half that, so 10 pt text lands near 14 px on screen.
FIG_WIDTH = 8.0  # [in]
DPI = 200
WIDTH_PX = int(FIG_WIDTH * DPI)
MAX_BYTES = 2_500_000
MAX_SECONDS = 8.0
GIF_COLOURS = 64

# Physical constants of water at 20 °C, as in the jbubble presets.
P_AMB = 101325.0  # [Pa]
RHO_L = 998.0  # [kg/m^3]
C_L = 1500.0  # [m/s]
MU = 1e-3  # [Pa s]
SIGMA_WATER = 0.072  # [N/m]

# ── Themes ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Theme:
    """A `jbubble.style` sheet plus the colours that the sheet doesn't set."""

    name: str
    drive: str  # acoustic drive trace, from the style sheet's comments
    ramp: tuple[str, ...]  # sequential map, from the surface to high values


# A navy-azure-aqua ramp whose lightness is monotone. Low values sit near the
# surface colour of each theme, so empty regions of a map read as background.
THEMES = (
    Theme(
        "light",
        drive="#8c959f",
        ramp=(
            "#f4f9fd",
            "#cfe6f6",
            "#8cc6ea",
            "#3d9bd6",
            "#0d77ca",
            "#0b56a0",
            "#0a3a73",
            "#08254d",
        ),
    ),
    Theme(
        "dark",
        drive="#6e7681",
        ramp=(
            "#0d1117",
            "#0b2340",
            "#0b3d6b",
            "#0d5e9c",
            "#1f84c9",
            "#45a9e0",
            "#86d0ee",
            "#d3f1fa",
        ),
    ),
)

# Slightly larger text than the style sheets, for display at half size.
RC = {
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.dpi": DPI,
    "savefig.bbox": "standard",
}


@dataclass(frozen=True)
class Palette:
    """Colours of the active theme, read from Matplotlib's `rcParams`."""

    name: str
    surface: str
    ink: str
    ink2: str
    rule: str
    series: tuple[str, ...]
    drive: str
    cmap: LinearSegmentedColormap


@contextlib.contextmanager
def themed(theme: Theme) -> Iterator[Palette]:
    """Apply `jbubble.style.<theme>` and yield its palette."""
    with plt.style.context(f"jbubble.style.{theme.name}"), plt.rc_context(RC):
        rc = plt.rcParams
        yield Palette(
            name=theme.name,
            surface=rc["figure.facecolor"],
            ink=rc["text.color"],
            ink2=rc["xtick.labelcolor"],
            rule=rc["axes.edgecolor"],
            series=tuple(rc["axes.prop_cycle"].by_key()["color"]),
            drive=theme.drive,
            cmap=LinearSegmentedColormap.from_list(f"jb_ramp_{theme.name}", theme.ramp),
        )


# ── Output ─────────────────────────────────────────────────────────────────────


def write_gif(
    fig: plt.Figure,
    update: Callable[[int], None],
    frames: Sequence[int],
    path: pathlib.Path,
    fps: int,
    colours: int = GIF_COLOURS,
) -> pathlib.Path:
    """Render `update(frame)` for each frame and encode a looping GIF.

    The first frame fixes the layout, so axes don't jump when text changes.
    ffmpeg (from imageio-ffmpeg) builds one palette for the whole animation and
    maps every frame to it without dithering, which keeps flat line art small.
    """
    update(frames[0])
    fig.canvas.draw()
    fig.set_layout_engine("none")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(pathlib.Path(tmp) / "f_%04d.png")
        for i, frame in enumerate(frames):
            update(frame)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            Image.fromarray(rgba[..., :3]).save(pattern % i, compress_level=1)
        palette = str(pathlib.Path(tmp) / "palette.png")
        common = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            pattern,
        ]
        subprocess.run(
            [
                *common,
                "-vf",
                f"palettegen=max_colors={colours}:stats_mode=full",
                palette,
            ],
            check=True,
        )
        subprocess.run(
            [
                *common,
                "-i",
                palette,
                "-lavfi",
                "paletteuse=dither=none:diff_mode=rectangle",
                "-loop",
                "0",
                str(path),
            ],
            check=True,
        )
    plt.close(fig)
    return path


def write_png(fig: plt.Figure, path: pathlib.Path) -> pathlib.Path:
    """Save a still at the asset width, on the theme's solid background."""
    fig.savefig(path, dpi=DPI, bbox_inches=None, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def held(n_frames: int, hold: int) -> list[int]:
    """Frame order that opens and closes on the last frame.

    The loop wraps from the closing hold into the opening one, so readers who
    see only the first frame see the finished result.
    """
    last = n_frames - 1
    return [last] * hold + list(range(n_frames)) + [last] * hold


# ── Asset 1: an oscillating bubble beside its radius trace ─────────────────────


def hero_bubble(out: pathlib.Path, quick: bool) -> list[pathlib.Path]:
    """A free microbubble driven into nonlinear oscillation, drawn to scale."""
    R0, freq, pressure, cycles = 2e-6, 1e6, 120e3, 6
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=SIGMA_WATER),
        medium=NewtonianMedium(mu=MU),
        R0=R0,
        P_amb=P_AMB,
        rho_L=RHO_L,
        c_L=C_L,
    )
    pulse = ToneBurst(
        freq=freq,
        pressure=pressure,
        shape=Sine(),
        cycle_num=cycles,
        envelope=HannEnvelope(),
    )
    res = run_simulation(eom, pulse, save_spec=SaveSpec(2000), t_max=10e-6)
    assert bool(res.converged)
    t = np.asarray(res.ts) * 1e6  # [µs]
    r = np.asarray(res.radius) * 1e6  # [µm]
    p = np.asarray(res.driving_pressure) / 1e3  # [kPa]
    r0 = R0 * 1e6
    print(f"  R/R0 from {r.min() / r0:.2f} to {r.max() / r0:.2f}")

    n_frames, fps = (40, 20) if quick else (150, 20)
    idx = np.linspace(0, len(t) - 1, n_frames).round().astype(int)
    # Start the loop at the largest expansion: the first frame is the poster.
    start = int(np.argmin(np.abs(idx - np.argmax(r))))
    order = [(start + k) % n_frames for k in range(n_frames)]

    paths = []
    for theme in THEMES:
        with themed(theme) as pal:
            c0 = pal.series[0]
            fig = plt.figure(figsize=(FIG_WIDTH, 3.3))
            gs = fig.add_gridspec(
                2, 2, width_ratios=[1.0, 2.5], height_ratios=[2.2, 1.0]
            )
            ax_b = fig.add_subplot(gs[:, 0])
            ax_r = fig.add_subplot(gs[0, 1])
            ax_p = fig.add_subplot(gs[1, 1], sharex=ax_r)

            # The bubble, in micrometres, with the equilibrium radius dashed.
            lim = 1.12 * r.max()
            ax_b.set_xlim(-lim, lim)
            ax_b.set_ylim(-lim, 1.45 * lim)  # room for the readout above the bubble
            ax_b.set_aspect("equal")
            ax_b.axis("off")
            ax_b.add_patch(
                Circle((0, 0), r0, fill=False, lw=1.0, ls=(0, (3, 2)), ec=pal.ink2)
            )
            body = Circle((0, 0), r0, fc=c0, ec="none", alpha=0.25)
            rim = Circle((0, 0), r0, fill=False, ec=c0, lw=2.2)
            for patch in (body, rim):
                ax_b.add_patch(patch)
            bar_y = -0.97 * lim
            ax_b.plot([-lim, -lim + 1.0], [bar_y, bar_y], color=pal.ink, lw=2.0)
            ax_b.text(
                -lim + 0.5, bar_y + 0.08 * lim, "1 µm", ha="center", color=pal.ink
            )
            ax_b.text(
                lim,
                bar_y,
                "dashed: R₀",
                ha="right",
                va="center",
                color=pal.ink2,
                fontsize=9,
            )
            label = ax_b.text(-lim, 1.45 * lim, "", ha="left", va="top", color=pal.ink)

            # R(t): the part already played in full colour, the rest faded.
            ax_r.axhline(r0, color=pal.ink2, lw=0.8, ls=(0, (3, 2)))
            (future,) = ax_r.plot([], [], color=c0, lw=1.6, alpha=0.3)
            (past,) = ax_r.plot([], [], color=c0, lw=2.0)
            (dot,) = ax_r.plot([], [], "o", ms=7, mfc=c0, mec=pal.surface, mew=1.8)
            ax_r.set_ylabel("R (µm)")
            ax_r.set_ylim(0, 1.08 * r.max())
            ax_r.set_title(
                f"Free {r0:.0f} µm bubble, {freq / 1e6:.0f} MHz, "
                f"{pressure / 1e3:.0f} kPa (Keller–Miksis)"
            )
            ax_r.tick_params(labelbottom=False)

            ax_p.plot(t, p, color=pal.drive, lw=1.4)
            ax_p.set_ylabel("drive (kPa)")
            ax_p.set_xlabel("time (µs)")
            ax_p.set_xlim(t[0], t[-1])
            cursors = [ax.axvline(0, color=pal.ink2, lw=0.8) for ax in (ax_r, ax_p)]

            def update(
                k: int, artists=(body, rim, label, future, past, dot, cursors)
            ) -> None:
                body, rim, label, future, past, dot, cursors = artists
                i = idx[k]
                body.set_radius(r[i])
                rim.set_radius(r[i])
                label.set_text(f"R = {r[i]:.2f} µm,  t = {t[i]:.2f} µs")
                past.set_data(t[: i + 1], r[: i + 1])
                future.set_data(t[i:], r[i:])
                dot.set_data([t[i]], [r[i]])
                for c in cursors:
                    c.set_xdata([t[i], t[i]])

            paths.append(
                write_gif(fig, update, order, out / f"hero-bubble-{pal.name}.gif", fps)
            )
    return paths


# ── Asset 2: gradient ascent on a GridSweep response map ───────────────────────

CLIMB_P0, CLIMB_FC, CLIMB_BW = 80e3, 1.5e6, 0.45e6  # transducer passband
CLIMB_F = (0.5e6, 3.0e6)  # [Hz]
CLIMB_R = (1.0e-6, 5.0e-6)  # [m]


def _climb_sim(freq, R0):
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=SIGMA_WATER),
        medium=NewtonianMedium(mu=MU),
        R0=R0,
        P_amb=P_AMB,
        rho_L=RHO_L,
        c_L=C_L,
    )
    # The transducer delivers less pressure away from its centre frequency, so
    # the map has one peak instead of an unbounded resonance ridge.
    pressure = CLIMB_P0 * jnp.exp(-0.5 * ((freq - CLIMB_FC) / CLIMB_BW) ** 2)
    pulse = ToneBurst(
        freq=freq, pressure=pressure, shape=Sine(), cycle_num=6, envelope=HannEnvelope()
    )
    return run_simulation(eom, pulse, save_spec=SaveSpec(256), t_max=8e-6)


def _peak_expansion(freq, R0):
    """A smooth maximum of R/R0: differentiable, and close to the true peak."""
    x = _climb_sim(freq, R0).radius / R0
    beta = 50.0
    return (jax.nn.logsumexp(beta * x) - jnp.log(x.shape[0])) / beta


def gradient_climb(out: pathlib.Path, quick: bool) -> list[pathlib.Path]:
    """`jax.grad` and optax climb to the resonance peak of a sweep map."""
    n_grid, n_steps = (20, 12) if quick else (72, 45)
    freqs = jnp.linspace(*CLIMB_F, n_grid)
    radii = jnp.linspace(*CLIMB_R, n_grid)
    sweep = GridSweep(
        fn=_peak_expansion, search_space={"R0": radii, "freq": freqs}, progress=False
    )
    grid = np.asarray(sweep.run())  # axes in sorted order: (R0, freq)
    print(
        f"  map of {grid.size} simulations, peak R/R0 {grid.min():.2f} to {grid.max():.2f}"
    )

    # Climb in unconstrained coordinates u: a sigmoid maps each one onto the
    # map's range on a log scale, so every step stays inside the map.
    (f_lo, f_hi), (r_lo, r_hi) = CLIMB_F, CLIMB_R

    def to_physical(u):
        s = jax.nn.sigmoid(u)
        return f_lo * (f_hi / f_lo) ** s[0], r_lo * (r_hi / r_lo) ** s[1]

    def from_physical(f, r):
        s = jnp.array(
            [
                jnp.log(f / f_lo) / jnp.log(f_hi / f_lo),
                jnp.log(r / r_lo) / jnp.log(r_hi / r_lo),
            ]
        )
        return jnp.log(s / (1 - s))

    value_and_grad = jax.jit(
        jax.value_and_grad(lambda u: -_peak_expansion(*to_physical(u)))
    )
    optimiser = optax.adam(0.12)
    u = from_physical(2.7e6, 4.2e-6)
    state = optimiser.init(u)
    path, values = [], []
    for _ in range(n_steps):
        loss, g = value_and_grad(u)
        path.append([float(x) for x in to_physical(u)])
        values.append(-float(loss))
        updates, state = optimiser.update(g, state)
        u = optax.apply_updates(u, updates)
    path = np.array(path)
    print(
        f"  climb: {path[0, 0] / 1e6:.2f} MHz, {path[0, 1] * 1e6:.2f} µm (R/R0 {values[0]:.2f})"
        f" -> {path[-1, 0] / 1e6:.2f} MHz, {path[-1, 1] * 1e6:.2f} µm (R/R0 {values[-1]:.2f})"
    )
    traces = jax.jit(jax.vmap(lambda f, r: _climb_sim(f, r).radius / r))(
        jnp.asarray(path[:, 0]), jnp.asarray(path[:, 1])
    )
    traces = np.asarray(traces)
    ts = np.asarray(_climb_sim(CLIMB_FC, 2e-6).ts) * 1e6

    frames = held(n_steps, hold=6)
    paths = []
    for theme in THEMES:
        with themed(theme) as pal:
            fig, (ax_m, ax_r) = plt.subplots(
                1,
                2,
                figsize=(FIG_WIDTH, 3.4),
                gridspec_kw={"width_ratios": [1.15, 1.0]},
            )
            fig.get_layout_engine().set(wspace=0.12)
            mesh = ax_m.pcolormesh(
                np.asarray(freqs) / 1e6,
                np.asarray(radii) * 1e6,
                grid,
                shading="gouraud",
                cmap=pal.cmap,
                rasterized=True,
            )
            bar = fig.colorbar(mesh, ax=ax_m, pad=0.02, fraction=0.06)
            bar.outline.set_visible(False)
            bar.set_label("peak R / R₀")
            ax_m.grid(False)
            ax_m.set_xlabel("drive frequency (MHz)")
            ax_m.set_ylabel("R₀ (µm)")
            ax_m.set_title(f"GridSweep map, {grid.size:,} simulations")
            accent = pal.series[1]
            ax_m.plot(
                path[0, 0] / 1e6,
                path[0, 1] * 1e6,
                "o",
                ms=6,
                mfc="none",
                mec=accent,
                mew=1.6,
            )
            (trail,) = ax_m.plot([], [], color=accent, lw=2.0, label="jax.grad + Adam")
            (head,) = ax_m.plot([], [], "o", ms=8, mfc=accent, mec=pal.surface, mew=1.8)
            ax_m.legend(loc="lower right")

            ax_r.axhline(1.0, color=pal.ink2, lw=0.8, ls=(0, (3, 2)))
            (line,) = ax_r.plot([], [], color=pal.series[0], lw=2.0)
            ax_r.set_xlabel("time (µs)")
            ax_r.set_ylabel("R / R₀")
            ax_r.set_xlim(ts[0], ts[-1])
            ax_r.set_ylim(0, 1.1 * traces.max())
            title = ax_r.set_title(" ")

            def update(k: int, artists=(trail, head, line, title)) -> None:
                trail, head, line, title = artists
                trail.set_data(path[: k + 1, 0] / 1e6, path[: k + 1, 1] * 1e6)
                head.set_data([path[k, 0] / 1e6], [path[k, 1] * 1e6])
                line.set_data(ts, traces[k])
                title.set_text(f"Step {k + 1}: peak R/R₀ = {values[k]:.2f}")

            paths.append(
                # More colours than the default, so the map's gradient doesn't band.
                write_gif(
                    fig,
                    update,
                    frames,
                    out / f"gradient-climb-{pal.name}.gif",
                    fps=8,
                    colours=128,
                )
            )
    return paths


# ── Asset 3: a neural network learns the shell law sigma(R) ────────────────────

SIGMA_R0 = 2.5e-6  # [m]
SIGMA_MAX = 0.072  # [N/m], the rupture surface tension and the network's bound
SIGMA_T_MAX = 10e-6  # [s]
SIGMA_SAVE = SaveSpec(256)
SIGMA_PRESSURES = (30e3, 60e3, 90e3, 120e3, 150e3, 180e3)  # [Pa]
SIGMA_SHOWN = (1, 5)  # pressures drawn in the R(t) panels


def _shell_model(sigma, pressure):
    eom = RayleighPlesset(
        gas=PolytropicGas(gamma=1.07),
        shell=LipidShell(sigma=sigma, kappa_s=5e-9),
        medium=NewtonianMedium(mu=MU),
        R0=SIGMA_R0,
        P_amb=P_AMB,
        rho_L=RHO_L,
    )
    pulse = ToneBurst(
        freq=1e6, pressure=pressure, shape=Sine(), cycle_num=5, envelope=HannEnvelope()
    )
    return eom, pulse


def neural_sigma(out: pathlib.Path, quick: bool) -> list[pathlib.Path]:
    """Learn sigma(R) from R(t) at six pressures, against the Marmottant truth."""
    n_steps, n_snapshots = (40, 12) if quick else (3000, 50)
    truth = MarmottantSurfaceTension(
        R_buckle_ratio=0.99, chi=0.5, sigma_rupture=SIGMA_MAX
    )

    def simulate(sigma, pressure):
        return run_simulation(
            *_shell_model(sigma, pressure), save_spec=SIGMA_SAVE, t_max=SIGMA_T_MAX
        )

    conditions = [
        {"pressure": p, "radius": simulate(truth, p).radius} for p in SIGMA_PRESSURES
    ]

    # sigma(R/R0) from a small MLP, bounded to (0, SIGMA_MAX) by its final
    # activation, so the shell law stays physical at every step.
    net = eqx.nn.MLP(
        in_size=1,
        out_size=1,
        width_size=4,
        depth=2,
        final_activation=lambda x: SIGMA_MAX * jax.nn.sigmoid(x),
        key=jax.random.PRNGKey(0),
    )
    # Snapshots on a geometric schedule: early steps change the most.
    snap_steps = set(
        np.unique(np.geomspace(1, n_steps, n_snapshots).round().astype(int))
    )
    snap_steps |= {0, n_steps}
    snapshots: dict[int, NeuralProperty] = {}
    losses: list[float] = []

    def record(step: int, params: NeuralProperty, loss: float) -> None:
        losses.append(loss)
        if step in snap_steps:
            snapshots[step] = params

    t0 = time.perf_counter()
    fit = fit_parameters(
        lambda sigma, c: _shell_model(sigma, c["pressure"]),
        NeuralProperty(net=net),
        conditions=conditions,
        loss_fn=lambda res, c: normalised_mse_radius(res.radius, c["radius"], SIGMA_R0),
        optimizer=optax.adam(0.01),
        n_steps=n_steps,
        save_spec=SIGMA_SAVE,
        t_max=SIGMA_T_MAX,
        step_callback=record,
        log_every=0,
    )
    print(
        f"  {n_steps} steps in {time.perf_counter() - t0:.0f} s: loss {losses[0]:.2e} -> "
        f"{losses[-1]:.2e}, {fit.num_rejected} rejected"
    )

    steps = sorted(snapshots)
    xs = np.linspace(0.4, 1.6, 400)  # R/R0

    def sigma_curve(prop):
        state = lambda x: BubbleState(R=x * SIGMA_R0, R0=jnp.asarray(SIGMA_R0))  # noqa: E731
        return (
            np.asarray(jax.vmap(lambda x: prop(state(x)))(jnp.asarray(xs))) * 1e3
        )  # [mN/m]

    shown = jnp.asarray([SIGMA_PRESSURES[i] for i in SIGMA_SHOWN])

    @eqx.filter_jit
    def trace(sigma, pressures):
        return jax.vmap(lambda p: simulate(sigma, p).radius / SIGMA_R0)(pressures)

    sig_true = sigma_curve(truth)
    sig_fit = [sigma_curve(snapshots[s]) for s in steps]
    r_fit = [np.asarray(trace(snapshots[s], shown)) for s in steps]
    r_data = [np.asarray(conditions[i]["radius"]) / SIGMA_R0 for i in SIGMA_SHOWN]
    ts = np.linspace(0.0, SIGMA_T_MAX, SIGMA_SAVE.num_samples) * 1e6
    rms = float(np.sqrt(np.mean((sig_fit[-1] - sig_true) ** 2)))
    print(f"  sigma(R) RMS error over R/R0 in [0.4, 1.6]: {rms:.1f} mN/m")
    loss_steps = np.arange(len(losses))

    frames = held(len(steps), hold=5)
    paths = []
    for theme in THEMES:
        with themed(theme) as pal:
            c_model = pal.series[0]
            fig = plt.figure(figsize=(FIG_WIDTH, 3.4))
            gs = fig.add_gridspec(2, 3, width_ratios=[0.8, 1.2, 1.15])
            ax_l = fig.add_subplot(gs[:, 0])
            ax_r = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
            ax_s = fig.add_subplot(gs[:, 2])

            ax_l.plot(loss_steps[1:], losses[1:], color=pal.rule, lw=1.0)
            (loss_line,) = ax_l.plot([], [], color=c_model, lw=2.0)
            (loss_dot,) = ax_l.plot(
                [], [], "o", ms=7, mfc=c_model, mec=pal.surface, mew=1.8
            )
            ax_l.set_xscale("log")
            ax_l.set_yscale("log")
            ax_l.set_xlabel("training step")
            ax_l.set_title("Loss")

            lines = []
            for ax, data, i in zip(ax_r, r_data, SIGMA_SHOWN, strict=True):
                ax.plot(ts, data, "o", ms=2.0, color=pal.ink2, alpha=0.8, mec="none")
                (ln,) = ax.plot([], [], color=c_model, lw=1.8)
                lines.append(ln)
                pad = 0.12 * (data.max() - data.min())
                ax.set_ylim(data.min() - pad, data.max() + pad)
                ax.set_xlim(0, ts[-1])
                ax.set_ylabel("R / R₀")
                ax.text(
                    0.98,
                    0.95,
                    f"{SIGMA_PRESSURES[i] / 1e3:.0f} kPa",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    color=pal.ink2,
                )
            ax_r[0].tick_params(labelbottom=False)
            ax_r[0].set_title("R(t): data (dots) and model")
            ax_r[1].set_xlabel("time (µs)")

            ax_s.plot(
                xs,
                sig_true,
                color=pal.ink,
                lw=1.3,
                ls=(0, (4, 3)),
                label="Marmottant\n(truth)",
            )
            (sig_line,) = ax_s.plot([], [], color=c_model, lw=2.4, label="network")
            ax_s.set_xlim(xs[0], xs[-1])
            ax_s.set_ylim(-4, 1.1 * SIGMA_MAX * 1e3)
            ax_s.set_xlabel("R / R₀")
            ax_s.set_ylabel("σ (mN/m)")
            title = ax_s.set_title(" ")
            ax_s.legend(loc="center left")

            def update(
                k: int, artists=(loss_line, loss_dot, lines, sig_line, title)
            ) -> None:
                loss_line, loss_dot, lines, sig_line, title = artists
                step = steps[k]
                shown_steps = loss_steps[1 : step + 1]
                loss_line.set_data(shown_steps, losses[1 : step + 1])
                loss_dot.set_data([max(step, 1)], [losses[step]])
                for ln, r in zip(lines, r_fit[k], strict=True):
                    ln.set_data(ts, r)
                sig_line.set_data(xs, sig_fit[k])
                title.set_text(f"σ(R) at step {step:,}")

            paths.append(
                write_gif(
                    fig, update, frames, out / f"neural-sigma-{pal.name}.gif", fps=8
                )
            )
    return paths


# ── Asset 4: emission spectrum vs drive pressure ───────────────────────────────

SPEC_R0, SPEC_F, SPEC_CYCLES = 2e-6, 1e6, 20
SPEC_T_MAX = 24e-6  # [s], the 20 µs burst plus 4 µs of ringing
SPEC_DISTANCE = 0.01  # [m], hydrophone distance
SPEC_F_SHOW = 5.5e6  # [Hz], top of the plotted band
BROADBAND_THRESHOLD = -10.0  # [dB re total]: a tenth of the power between lines
HARMONICS = (
    (0.5, "f/2"),
    (1, "f"),
    (1.5, "3f/2"),
    (2, "2f"),
    (3, "3f"),
    (4, "4f"),
    (5, "5f"),
)


def _emission_map(quick: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return frequencies [Hz], pressures [kPa], power re max, and R_max/R0."""
    n_pressures, n_samples = (24, 8192) if quick else (128, 16384)
    pressures = jnp.linspace(10e3, 400e3, n_pressures)
    eom = KellerMiksis(
        gas=VanDerWaalsGas(gamma=1.4, h_frac=1 / 8.86),
        shell=NoShell(sigma=SIGMA_WATER),
        medium=NewtonianMedium(mu=MU),
        R0=SPEC_R0,
        P_amb=P_AMB,
        rho_L=RHO_L,
        c_L=C_L,
    )
    dt = SPEC_T_MAX / (n_samples - 1)
    n_fft = 4 * n_samples  # zero padding interpolates the plotted spectrum
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    n_band = int(np.sum(freqs <= SPEC_F_SHOW))

    def spectrum(pressure):
        pulse = ToneBurst(
            freq=SPEC_F,
            pressure=pressure,
            shape=Sine(),
            cycle_num=SPEC_CYCLES,
            envelope=HannEnvelope(),
        )
        res = run_simulation(
            eom, pulse, save_spec=SaveSpec(n_samples), t_max=SPEC_T_MAX
        )
        # The monopole pressure is (rho_L / r) d(R^2 dR/dt)/dt, so its average
        # over each sample interval is a difference of R^2 dR/dt. Averaging is
        # an anti-aliasing filter: an inertial collapse lasts under a
        # nanosecond, and point samples of it alias into the plotted band.
        q = RHO_L / SPEC_DISTANCE * res.radius**2 * res.radial_velocity
        p_rad = jnp.diff(q) / dt  # [Pa]
        x = (p_rad - p_rad.mean()) * jnp.hanning(n_samples - 1)
        power = jnp.abs(jnp.fft.rfft(x, n=n_fft)[:n_band] * dt) ** 2
        return {
            "power": power,
            "expansion": res.radius.max() / SPEC_R0,
            "converged": res.converged,
        }

    sweep = GridSweep(fn=spectrum, search_space={"pressure": pressures}, progress=False)
    result = sweep.run()
    assert bool(np.all(result["converged"]))
    power = result["power"] / result["power"].max()
    return freqs[:n_band], np.asarray(pressures) / 1e3, power, result["expansion"]


def emission_spectrum(out: pathlib.Path, quick: bool) -> list[pathlib.Path]:
    """Harmonics, then broadband noise, as the drive passes the inertial threshold."""
    freqs, p_kpa, power, expansion = _emission_map(quick)

    # Broadband index: the share of the power above 0.2 f that lies between
    # the lines at multiples of f/2, more than 0.2 f from every line. Periodic
    # oscillation keeps it small; aperiodic inertial collapses fill the gaps.
    nearest = np.round(freqs / (SPEC_F / 2)) * (SPEC_F / 2)
    above = freqs > 0.2 * SPEC_F
    between = above & (np.abs(freqs - nearest) > 0.2 * SPEC_F)
    broadband = 10 * np.log10(
        power[:, between].mean(axis=1) / power[:, above].mean(axis=1)
    )
    onset = p_kpa[np.argmax(broadband > BROADBAND_THRESHOLD)]
    inertial = p_kpa[np.argmax(expansion >= 2.0)]
    # Above the onset, the response is chaotic: a few drives give a much
    # smaller collapse than their neighbours and show as dark rows. They're
    # features of the model, not solver error, and persist at rtol = 1e-9.
    # For display, average the power over 0.1 MHz: a chaotic signal's spectrum
    # has deep, narrow nulls that would otherwise streak the map.
    width = int(round(0.1 * SPEC_F / (freqs[1] - freqs[0])))
    kernel = np.ones(width) / width
    smooth = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 1, power
    )
    level = 10 * np.log10(smooth / smooth.max() + 1e-30)  # [dB re map maximum]
    print(
        f"  broadband onset at {onset:.0f} kPa; R_max first reaches 2 R0 at {inertial:.0f} kPa"
    )

    paths = []
    for theme in THEMES:
        with themed(theme) as pal:
            fig, (ax_m, ax_b) = plt.subplots(
                1,
                2,
                figsize=(FIG_WIDTH, 3.6),
                sharey=True,
                gridspec_kw={"width_ratios": [3.2, 1.0]},
            )
            mesh = ax_m.pcolormesh(
                freqs / 1e6,
                p_kpa,
                level,
                shading="gouraud",
                cmap=pal.cmap,
                vmin=-60,
                vmax=0,
                rasterized=True,
            )
            bar = fig.colorbar(mesh, ax=ax_m, pad=0.015, fraction=0.05)
            bar.outline.set_visible(False)
            bar.set_label("level (dB re max)")
            ax_m.grid(False)
            ax_m.set_xlabel("frequency (MHz)")
            ax_m.set_ylabel("drive pressure (kPa)")
            ax_m.set_title(
                "Spectrum of the radiated pressure, 2 µm bubble at 1 MHz", pad=20
            )
            for h, name in HARMONICS:
                ax_m.text(
                    h,
                    1.01,
                    name,
                    transform=ax_m.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    color=pal.ink2,
                    fontsize=9,
                )
            accent = pal.series[1]
            for ax in (ax_m, ax_b):
                ax.axhline(onset, color=accent, lw=1.2, ls=(0, (4, 3)))
            ax_m.text(
                0.02,
                onset,
                f"broadband onset, {onset:.0f} kPa",
                transform=ax_m.get_yaxis_transform(),
                ha="left",
                va="bottom",
                color=accent,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": pal.surface,
                    "ec": "none",
                    "alpha": 0.85,
                },
            )

            ax_b.axvline(BROADBAND_THRESHOLD, color=pal.ink2, lw=0.8, ls=(0, (2, 2)))
            ax_b.plot(broadband, p_kpa, color=pal.series[0], lw=1.8)
            ax_b.set_xlabel("between lines (dB)")
            ax_b.set_title("Broadband", pad=20)
            ax_b.tick_params(labelleft=False)
            ax_b.set_ylim(p_kpa[0], p_kpa[-1])
            paths.append(write_png(fig, out / f"emission-spectrum-{pal.name}.png"))
    return paths


# ── Manifest and budget ────────────────────────────────────────────────────────

ASSETS: dict[str, Callable[[pathlib.Path, bool], list[pathlib.Path]]] = {
    "hero-bubble": hero_bubble,
    "gradient-climb": gradient_climb,
    "neural-sigma": neural_sigma,
    "emission-spectrum": emission_spectrum,
}


def input_hash() -> str:
    """Hash this script, the jbubble package, and the lock file."""
    digest = hashlib.sha256()
    files = [pathlib.Path(__file__).resolve(), ROOT / "uv.lock"]
    package = ROOT / "jbubble"
    files += sorted(f for f in package.rglob("*") if f.suffix in {".py", ".mplstyle"})
    for f in files:
        if f.is_file():
            digest.update(f.relative_to(ROOT).as_posix().encode())
            digest.update(f.read_bytes())
    return digest.hexdigest()


def git_sha() -> str:
    """Return the commit that the assets came from, marked `+dirty` if needed."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}+dirty" if dirty else sha


def describe(path: pathlib.Path) -> dict[str, float | int]:
    """Measure one asset and check it against the budget."""
    with Image.open(path) as im:
        frames = getattr(im, "n_frames", 1)
        width, height = im.size
        duration = 0.0
        if frames > 1:
            for k in range(frames):
                im.seek(k)
                duration += im.info.get("duration", 0) / 1000
    size = path.stat().st_size
    problems = []
    if size > MAX_BYTES:
        problems.append(f"{size / 1e6:.2f} MB > {MAX_BYTES / 1e6:.1f} MB")
    if width != WIDTH_PX:
        problems.append(f"width {width} px != {WIDTH_PX} px")
    if duration > MAX_SECONDS:
        problems.append(f"{duration:.1f} s > {MAX_SECONDS:.0f} s")
    if problems:
        raise SystemExit(f"{path.name} is over budget: {'; '.join(problems)}")
    return {
        "bytes": size,
        "width": width,
        "height": height,
        "frames": frames,
        "duration_s": round(duration, 2),
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=sorted(ASSETS),
        default=list(ASSETS),
        help="assets to render",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="shrink every computation, for a smoke test",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=f"output directory (default: {DEFAULT_OUT.relative_to(ROOT)}, "
        "or a temporary directory with --quick)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    matplotlib.use("Agg")  # render off screen, whatever the default backend
    with contextlib.ExitStack() as stack:
        if args.out is not None:
            out = args.out
        elif args.quick:
            out = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory()))
        else:
            out = DEFAULT_OUT
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "manifest.json"
        manifest = {"assets": {}}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        provenance = {
            "input_hash": input_hash(),
            "jbubble_version": jbubble.__version__,
            "git_sha": git_sha(),
            "quick": args.quick,
        }
        for name in args.only:
            print(f"{name}:", flush=True)
            t0 = time.perf_counter()
            for path in ASSETS[name](out, args.quick):
                info = describe(path)
                manifest["assets"][path.name] = {**info, **provenance}
                print(
                    f"  {path.name}: {info['bytes'] / 1e6:.2f} MB, {info['width']}x"
                    f"{info['height']} px, {info['frames']} frames, {info['duration_s']} s"
                )
            print(f"  done in {time.perf_counter() - t0:.0f} s", flush=True)
        manifest = {
            "generator": "scripts/make_readme_assets.py",
            "budget": {
                "max_bytes": MAX_BYTES,
                "width_px": WIDTH_PX,
                "max_seconds": MAX_SECONDS,
            },
            "assets": dict(sorted(manifest["assets"].items())),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {len(args.only)} assets and the manifest to {out}")


if __name__ == "__main__":
    main()
