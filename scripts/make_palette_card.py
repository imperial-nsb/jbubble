"""Render the jbubble palette card for each Matplotlib theme.

The card shows the eight categorical slots of `jbubble.style.light` and
`jbubble.style.dark`, four sample series with the drive trace on its own axes,
and the sequential colour map. To regenerate the committed images, run this
command from the repository root:

    uv run python scripts/make_palette_card.py

The script writes `docs/assets/palette-light.png` and
`docs/assets/palette-dark.png`. The curves are illustrative and don't come
from a simulation.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMES = ("azure", "coral", "teal", "plum", "green", "violet", "amber", "crimson")
# Secondary ink and the drive-trace grey; the style sheets list the same values.
NEUTRALS = {
    "light": {"ink2": "#59636e", "drive": "#8c959f"},
    "dark": {"ink2": "#9198a1", "drive": "#6e7681"},
}


def draw_card(theme: str) -> plt.Figure:
    neutral = NEUTRALS[theme]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(8.0, 4.6))
    grid = fig.add_gridspec(
        3, 2, height_ratios=(1.0, 1.35, 0.75), width_ratios=(2.2, 1.0)
    )
    fig.suptitle(f"jbubble figure palette, {theme} theme", x=0.02, ha="left")

    # Categorical slots: a swatch per slot, labelled in ink, never in the colour.
    ax = fig.add_subplot(grid[0, :])
    ax.set_axis_off()
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    for i, (color, name) in enumerate(zip(colors, NAMES, strict=True)):
        ax.add_patch(
            FancyBboxPatch(
                (i + 0.08, 0.46),
                0.84,
                0.5,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(i + 0.08, 0.3, f"C{i} {name}", fontsize=9, va="center")
        ax.text(i + 0.08, 0.08, color, fontsize=8, va="center", color=neutral["ink2"])

    # Four series: the first four slots stay distinct from one another.
    t = np.linspace(0.0, 6.0, 600)
    lines = fig.add_subplot(grid[1:, 0])
    for k in range(4):
        damping = 0.25 + 0.12 * k
        radius = 1.0 + (0.45 - 0.08 * k) * np.exp(-damping * t) * np.sin(
            2 * np.pi * (1.0 + 0.15 * k) * t
        )
        lines.plot(t, radius, label=f"C{k} {NAMES[k]}")
    lines.set_title("Four series")
    lines.set_xlabel(r"Time [$\mu$s]")
    lines.set_ylabel(r"$R / R_0$")
    lines.legend(loc="upper right", ncols=2)

    # The drive goes on its own axes in a neutral grey.
    drive = fig.add_subplot(grid[1, 1])
    drive.plot(
        t, np.sin(2 * np.pi * t) * np.exp(-((t - 3) ** 2)), color=neutral["drive"]
    )
    drive.set_title("Drive trace")
    drive.set_xlabel(r"Time [$\mu$s]")
    drive.set_yticks([])

    ramp = fig.add_subplot(grid[2, 1])
    ramp.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto")
    ramp.set_title(f"Sequential: {plt.rcParams['image.cmap']}")
    ramp.set_axis_off()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "docs" / "assets",
        help="directory for palette-light.png and palette-dark.png",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        with plt.style.context(f"jbubble.style.{theme}"):
            fig = draw_card(theme)
            path = args.out / f"palette-{theme}.png"
            # Pin the metadata so that unchanged cards produce identical files.
            fig.savefig(path, dpi=150, metadata={"Software": None})
            plt.close(fig)
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
