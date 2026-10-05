"""Griffe extensions for the jbubble API reference.

mkdocs.yml loads this file through the mkdocstrings `extensions` option.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import importlib
import inspect
import types
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


class FactoryDefaults(griffe.Extension):
    """Show the default that a dataclass field's factory makes.

    With `force_inspection: true`, griffe reads each class signature from
    `inspect.signature`, which shows the default of a field with a
    `default_factory` as `<factory>`, as in `envelope: Envelope = <factory>`.
    That hides the default, and it isn't valid Python, so ruff can't format
    the signature, which then stays on one long line. This extension replaces
    `<factory>` with code that makes the default:

    - A class, such as `default_factory=Sine`, becomes a call, `Sine()`, that
      links to the class.
    - A lambda, such as `default_factory=lambda: jnp.zeros(())`, becomes its
      body, `jnp.zeros(())`.
    - Another named callable, such as a function `make_default`, becomes a
      call, `make_default()`.
    - Anything else becomes `...`.
    """

    def on_class_members(
        self,
        *,
        node: Any,
        cls: griffe.Class,
        **kwargs: Any,
    ) -> None:
        if not isinstance(node, griffe.ObjectNode):
            return
        if not dataclasses.is_dataclass(node.obj):
            return
        init = cls.members.get("__init__")
        if not isinstance(init, griffe.Function):
            return
        factories = {
            field.name: field.default_factory
            for field in dataclasses.fields(node.obj)
            if field.default_factory is not dataclasses.MISSING
        }
        for parameter in init.parameters:
            if parameter.name in factories and str(parameter.default) == "<factory>":
                parameter.default = factory_default(factories[parameter.name])


def factory_default(factory: Any) -> str | griffe.Expr:
    """Return code that makes the default of a `default_factory`."""
    if isinstance(factory, type):
        return griffe.ExprCall(base_name(factory), [])
    name = getattr(factory, "__qualname__", "")
    if name.endswith("<lambda>"):
        return lambda_body(factory) or "..."
    if name.isidentifier():
        return f"{name}()"
    return "..."


def lambda_body(function: Any) -> str | None:
    """Return the source of a lambda's body, or `None` if it's ambiguous."""
    module = inspect.getmodule(function)
    if module is None:
        return None
    tree, source = parsed_source(module)
    if tree is None:
        return None
    line = function.__code__.co_firstlineno
    lambdas = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Lambda) and node.lineno == line
    ]
    if len(lambdas) != 1:  # several lambdas on one line
        return None
    body = ast.get_source_segment(source, lambdas[0].body)
    if body is None or "\n" in body:
        return ast.unparse(lambdas[0].body)
    return body


@functools.cache
def parsed_source(module: types.ModuleType) -> tuple[ast.Module | None, str]:
    """Parse a module's source once."""
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return None, ""
    return ast.parse(source), source


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
