"""Check that the docs site theme matches the jbubble figure palette.

`docs/stylesheets/extra.css` and `docs/assets/images/logo.svg` repeat colours
from the style sheets in `jbubble/style/`. These tests fail when they drift
apart, and when the theme's text colours lose the contrast that extra.css
promises.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSS = ROOT / "docs" / "stylesheets" / "extra.css"
LOGO = ROOT / "docs" / "assets" / "images" / "logo.svg"
pytestmark = pytest.mark.skipif(
    not CSS.exists(), reason="needs docs/ from a repository checkout"
)

# Zensical's inline-code backgrounds: #f5f5f5 in the light scheme, and
# hsla(var(--md-hue), 20%, 10%, 1) in the dark scheme, with the hue of 216
# that extra.css sets.
CODE_BACKGROUND = {"default": "#f5f5f5", "slate": "#14181f"}


def palette(theme):
    """Return the surface colour and the colour cycle of a style sheet."""
    text = (ROOT / "jbubble" / "style" / f"{theme}.mplstyle").read_text()
    (surface,) = re.findall(r"^figure\.facecolor:\s*([0-9a-f]{6})\s*$", text, re.M)
    (cycle,) = re.findall(r"^axes\.prop_cycle:.*\[(.*)\]", text, re.M)
    return f"#{surface}", [f"#{c}" for c in re.findall(r'"([0-9a-f]{6})"', cycle)]


def tokens(scheme):
    """Return the colour tokens that extra.css sets for a colour scheme."""
    (block,) = re.findall(
        rf'^\[data-md-color-scheme="{scheme}"\]\s*\{{(.*?)\}}',
        CSS.read_text(),
        re.M | re.S,
    )
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-f]{6}(?:[0-9a-f]{2})?);", block))


def rgb(color):
    return [int(color[i : i + 2], 16) for i in (1, 3, 5)]


def blend(color, background):
    """Flatten an #rrggbbaa colour onto an opaque background."""
    alpha = int(color[7:9], 16) / 255
    pairs = zip(rgb(color), rgb(background), strict=True)
    mixed = [round(a * alpha + b * (1 - alpha)) for a, b in pairs]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def contrast(foreground, background):
    """Return the WCAG 2 contrast ratio of two opaque colours."""

    def luminance(color):
        r, g, b = (
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in (v / 255 for v in rgb(color))
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    high, low = sorted([luminance(foreground), luminance(background)], reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_light_scheme_uses_the_light_palette():
    _, colors = palette("light")
    light = tokens("default")
    assert light["--md-primary-fg-color"] == colors[0]
    assert light["--md-accent-fg-color--transparent"][:7] == colors[1]


def test_dark_scheme_uses_the_dark_palette():
    surface, colors = palette("dark")
    dark = tokens("slate")
    assert dark["--md-default-bg-color"] == surface
    assert dark["--md-primary-bg-color"] == surface
    assert dark["--md-primary-fg-color"] == colors[0]
    assert dark["--md-typeset-a-color"] == colors[0]
    assert dark["--md-accent-fg-color"] == colors[1]
    assert dark["--md-accent-fg-color--transparent"][:7] == colors[1]


def test_gallery_thumbnails_fill_with_the_light_surface():
    # The example figures use the light style, so the letterbox around a
    # thumbnail that isn't 16:9 takes the light figure surface.
    surface, _ = palette("light")
    (rule,) = re.findall(
        r"^\.md-typeset \.grid\.cards img\s*\{(.*?)\}", CSS.read_text(), re.M | re.S
    )
    assert re.findall(r"background-color:\s*(#[0-9a-f]{6});", rule) == [surface]


def test_logo_uses_the_first_two_light_colours():
    _, colors = palette("light")
    fills = re.findall(r'fill="(#[0-9a-f]{6})"', LOGO.read_text())
    assert set(colors[:2]) <= set(fills)


@pytest.mark.parametrize(("scheme", "theme"), [("default", "light"), ("slate", "dark")])
def test_links_and_the_accent_keep_aa_contrast(scheme, theme):
    # Links appear on the page and on inline code; the accent colours hover
    # states and the active navigation entry, which sits on the accent tint.
    surface, _ = palette(theme)
    scheme_tokens = tokens(scheme)
    tint = blend(scheme_tokens["--md-accent-fg-color--transparent"], surface)
    link = scheme_tokens["--md-typeset-a-color"]
    accent = scheme_tokens["--md-accent-fg-color"]
    for foreground, background in [
        (link, surface),
        (link, CODE_BACKGROUND[scheme]),
        (accent, surface),
        (accent, CODE_BACKGROUND[scheme]),
        (accent, tint),
    ]:
        assert contrast(foreground, background) >= 4.5, (foreground, background)
