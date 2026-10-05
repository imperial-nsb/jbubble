"""Tests for scripts/griffe_extensions.py, which the API reference loads.

The extensions need griffe from the docs dependency group
(`uv sync --group docs`), so these tests skip themselves without it.
"""

import importlib.util
import inspect
import pathlib
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "griffe_extensions.py"
if not SCRIPT_PATH.exists():  # the sdist ships tests/ but not scripts/
    pytest.skip("needs scripts/ from a repository checkout", allow_module_level=True)
griffe = pytest.importorskip("griffe")

_spec = importlib.util.spec_from_file_location("griffe_extensions", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
griffe_extensions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(griffe_extensions)

SOURCE = '''\
"""A package for the griffe extension tests."""

import equinox as eqx

from jbubble.bubble.property import Property


class Model(eqx.Module):
    """A model with a float default."""

    eps: float = 1e-6


class Custom(Property):
    """A jbubble property."""

    def __call__(self, state):
        return state.R


class Pair(tuple):
    """A tuple."""
'''


@pytest.fixture
def load(tmp_path, monkeypatch):
    package = tmp_path / "griffe_ext_sample"
    package.mkdir()
    (package / "__init__.py").write_text(textwrap.dedent(SOURCE))
    monkeypatch.syspath_prepend(str(tmp_path))

    def _load(*extensions):
        try:
            return griffe.load(
                "griffe_ext_sample",
                search_paths=[str(tmp_path)],
                force_inspection=True,
                extensions=griffe.load_extensions(*extensions),
            )
        finally:
            sys.modules.pop("griffe_ext_sample", None)

    return _load


def test_float_defaults_lose_the_docstring_of_float(load):
    plain = load()
    assert plain["Model.eps"].docstring.value == inspect.getdoc(float)
    fixed = load(griffe_extensions.DropValueTypeDocstrings)
    assert fixed["Model.eps"].docstring is None
    # The class keeps its own docstring.
    assert fixed["Model"].docstring.value == "A model with a float default."


def test_bases_become_names_with_public_paths(load):
    plain = load()
    assert plain["Model"].bases == ["equinox._module._module.Module"]
    package = load(griffe_extensions.CrossReferenceBases)
    paths = {
        name: [base.canonical_path for base in package[name].bases]
        for name in ("Model", "Custom", "Pair")
    }
    assert paths == {
        "Model": ["equinox.Module"],
        # jbubble classes keep their defining module, where docs/api/ anchors them.
        "Custom": ["jbubble.bubble.property.Property"],
        "Pair": ["tuple"],
    }
    assert all(isinstance(base, griffe.ExprName) for base in package["Model"].bases)
