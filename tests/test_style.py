"""Tests for the jbubble.style Matplotlib style sheets."""

import pathlib
import shutil
import subprocess
import sys
import warnings
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
STYLE_FILES = ("jbubble/style/light.mplstyle", "jbubble/style/dark.mplstyle")

# Surface colour and the first two categorical slots of each theme.
THEMES = {
    "light": ("#ffffff", "#0d77ca", "#e35a2d"),
    "dark": ("#0d1117", "#3e96ea", "#e6653c"),
}


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_style_loads_by_dotted_name(theme):
    pytest.importorskip("matplotlib")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    surface, c0, c1 = THEMES[theme]
    # Matplotlib warns about unknown or invalid keys instead of raising.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with plt.style.context(f"jbubble.style.{theme}"):
            rc = dict(mpl.rcParams)
    assert to_hex(rc["figure.facecolor"]) == surface
    assert to_hex(rc["axes.facecolor"]) == surface
    colors = [to_hex(c) for c in rc["axes.prop_cycle"].by_key()["color"]]
    assert len(colors) == 8
    assert colors[:2] == [c0, c1]
    assert not rc["axes.spines.top"]
    assert not rc["axes.spines.right"]
    assert rc["axes.grid"]


def test_style_package_does_not_import_matplotlib():
    code = "import sys, jbubble.style; sys.exit('matplotlib' in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the wheel")
def test_style_files_ship_in_the_wheel(tmp_path):
    # About a second with a warm uv cache.
    subprocess.run(
        ["uv", "build", "--wheel", "--no-sources", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    (wheel,) = tmp_path.glob("jbubble-*.whl")
    names = set(zipfile.ZipFile(wheel).namelist())
    assert set(STYLE_FILES) <= names
