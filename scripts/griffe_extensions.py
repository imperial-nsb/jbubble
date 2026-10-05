"""Griffe extensions for the jbubble API reference.

mkdocs.yml loads this file through the mkdocstrings `extensions` option.
"""

from __future__ import annotations

import inspect
from typing import Any

import griffe


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
