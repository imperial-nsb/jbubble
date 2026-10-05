"""Griffe extensions for the jbubble API reference.

mkdocs.yml loads this file through the mkdocstrings `extensions` option.
"""

from __future__ import annotations

import collections
import importlib
import inspect
from typing import Any

import griffe

# The descriptor type of a namedtuple field, such as `BubblePreset.eom`.
_TUPLEGETTER = type(collections.namedtuple("_Pair", "first").first)


class DropValueTypeDocstrings(griffe.Extension):
    """Drop attribute docstrings that griffe copies from the value's type.

    With `force_inspection: true`, griffe reads each attribute's docstring from
    the attribute's value. For a class attribute such as `eps: float = 1e-6`,
    the value is a `float`, so the API reference would show `float.__doc__`
    ("Convert a string or number to a floating-point number, if possible.")
    under `eps`. This extension removes a docstring that matches the
    docstring of the value's type.
    """

    def on_attribute_instance(
        self,
        *,
        node: Any,
        attr: griffe.Attribute,
        **kwargs: Any,
    ) -> None:
        if attr.docstring is None or not isinstance(node, griffe.ObjectNode):
            return
        type_doc = inspect.getdoc(type(node.obj))
        if type_doc and attr.docstring.value.strip() == type_doc.strip():
            attr.docstring = None


class CrossReferenceBases(griffe.Extension):
    """Turn the base classes that griffe inspects into cross-references.

    With `force_inspection: true`, griffe records each base class as a plain
    string with the module that defines it, such as
    `equinox._module._module.Module`. The API reference then shows that private
    path as code, without a link. This extension replaces each base with a
    name that mkdocstrings links:

    - A jbubble class keeps the path of its defining module, which is where
      its `:::` directive in `docs/api/` puts its anchor.
    - A class from another package takes the shortest public path that
      exposes the same object, such as `equinox.Module`, which is the path in
      that package's inventory.
    - A built-in class drops the `builtins.` prefix, as in Python's inventory.
    """

    def on_class_instance(
        self,
        *,
        node: Any,
        cls: griffe.Class,
        **kwargs: Any,
    ) -> None:
        if not isinstance(node, griffe.ObjectNode):
            return
        bases = [base for base in node.obj.__bases__ if base is not object]
        if len(bases) != len(cls.bases):
            return
        cls.bases = [base_name(base) for base in bases]


class NamedTupleFields(griffe.Extension):
    """Drop the value and docstring that inspection gives `NamedTuple` fields.

    With `force_inspection: true`, each field of a `NamedTuple` class is a
    descriptor, so the API reference would show the field `eom` with the value
    `_tuplegetter(0, 'Alias for field number 0')` and the docstring "Alias for
    field number 0". Without them, a field renders as it does with static
    analysis, and the class docstring's `Attributes` section describes it.
    """

    def on_attribute_instance(
        self,
        *,
        node: Any,
        attr: griffe.Attribute,
        **kwargs: Any,
    ) -> None:
        if isinstance(node, griffe.ObjectNode) and type(node.obj) is _TUPLEGETTER:
            attr.value = None
            attr.docstring = None


def base_name(base: type) -> griffe.ExprName:
    """Return a name expression whose canonical path documents `base`."""
    module = base.__module__
    if module == "builtins":
        return griffe.ExprName(base.__qualname__)
    if module.partition(".")[0] != "jbubble":
        module = public_module(base)
    return griffe.ExprName(base.__qualname__, parent=module)


def public_module(obj: type) -> str:
    """Return the shortest module path that exposes `obj` under its name."""
    parts = obj.__module__.split(".")
    for end in range(1, len(parts)):
        prefix = ".".join(parts[:end])
        try:
            target: Any = importlib.import_module(prefix)
            for attribute in obj.__qualname__.split("."):
                target = getattr(target, attribute)
        except (ImportError, AttributeError):
            continue
        if target is obj:
            return prefix
    return obj.__module__
