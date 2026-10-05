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

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from jbubble.bubble.property import Property


def unit():
    return 1.0


class Model(eqx.Module):
    """A model with a float default."""

    eps: float = 1e-6


class Custom(Property):
    """A jbubble property."""

    def __call__(self, state):
        return state.R


class Pair(tuple):
    """A tuple."""


class Preset(NamedTuple):
    """A named pair."""

    model: Model
    scale: float


class Pulse(eqx.Module):
    """A pulse with factory defaults."""

    model: Model = eqx.field(default_factory=Model)
    offset: object = eqx.field(
        default_factory=lambda: jnp.zeros(()),
    )
    scale: float = eqx.field(default_factory=unit)
    gain: float = 2.0
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


def test_named_tuple_fields_lose_the_descriptor_value_and_docstring(load):
    plain = load()
    assert plain["Preset.model"].docstring.value == "Alias for field number 0"
    assert "_tuplegetter" in str(plain["Preset.model"].value)
    package = load(griffe_extensions.NamedTupleFields)
    for field in ("model", "scale"):
        assert package[f"Preset.{field}"].value is None
        assert package[f"Preset.{field}"].docstring is None


def test_factory_defaults_show_the_code_that_makes_them(load):
    def defaults(package):
        params = package["Pulse"].parameters
        return {p.name: p.default for p in params if p.name != "self"}

    plain = defaults(load())
    assert [str(value) for value in plain.values()] == ["<factory>"] * 3 + ["2.0"]
    fixed = defaults(load(griffe_extensions.FactoryDefaults))
    assert {name: str(value) for name, value in fixed.items()} == {
        "model": "Model()",
        "offset": "jnp.zeros(())",
        "scale": "unit()",
        "gain": "2.0",
    }
    # A class becomes a call that links to the class.
    assert isinstance(fixed["model"], griffe.ExprCall)
    assert fixed["model"].function.canonical_path == "griffe_ext_sample.Model"


def test_ambiguous_lambdas_default_to_an_ellipsis():
    # Two lambdas on one line: the extension can't tell which one is the factory.
    first, second = (lambda: 1), (lambda: 2)
    assert griffe_extensions.factory_default(first) == "..."
    assert griffe_extensions.factory_default(second) == "..."
