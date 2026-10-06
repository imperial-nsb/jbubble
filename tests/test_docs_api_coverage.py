"""Check that the API reference in docs/api/ covers the public API.

Each name in a public module's `__all__` needs a `::: dotted.path` directive
in a page under docs/api/, either for the object where it's defined or for
the module that exports it. A directive for a whole module covers every name
in that module's `__all__`.
"""

import importlib
import inspect
import pathlib
import pkgutil
import re

import jbubble
import pytest

DOCS_API = pathlib.Path(__file__).resolve().parents[1] / "docs" / "api"
pytestmark = [
    # The sdist ships tests/ but not docs/.
    pytest.mark.skipif(
        not DOCS_API.is_dir(), reason="needs docs/api/ from a repository checkout"
    ),
]
# The package that the optional `examples` extra installs. Both tests skip
# a module that fails to import because it is missing.
OPTIONAL = {"matplotlib"}


def missing_optional(error: ImportError) -> bool:
    """Return whether an import failed only because an optional package is missing.

    A module can re-raise the `ModuleNotFoundError` for an optional package as
    an `ImportError` with an install hint, so check the cause too.
    """
    return any(
        isinstance(cause, ModuleNotFoundError)
        and (cause.name or "").partition(".")[0] in OPTIONAL
        for cause in (error, error.__cause__)
    )


def public_modules():
    names = ["jbubble"] + [
        info.name
        for info in pkgutil.walk_packages(jbubble.__path__, prefix="jbubble.")
        if not any(part.startswith("_") for part in info.name.split("."))
    ]
    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            if missing_optional(error):
                continue
            raise
        yield module


def directives() -> set[str]:
    text = "\n".join(page.read_text() for page in sorted(DOCS_API.glob("*.md")))
    return set(re.findall(r"^:::[ \t]*([\w.]+)", text, flags=re.MULTILINE))


def accepted_paths(module, name: str) -> set[str]:
    """Return the directive paths that document `module.name`."""
    obj = getattr(module, name)
    paths = {f"{module.__name__}.{name}", module.__name__}
    if inspect.isclass(obj) or inspect.isfunction(obj):
        paths |= {f"{obj.__module__}.{obj.__qualname__}", obj.__module__}
    return paths


def test_every_public_name_has_a_directive():
    documented = directives()
    missing = sorted(
        f"{module.__name__}.{name}"
        for module in public_modules()
        for name in getattr(module, "__all__", [])
        if not name.startswith("__")
        and not inspect.ismodule(getattr(module, name))
        and not accepted_paths(module, name) & documented
    )
    assert not missing, "Add a ::: directive to docs/api/ for:\n" + "\n".join(missing)


def test_every_directive_names_an_object():
    stale = []
    for path in sorted(directives()):
        module_name, _, attr = path.rpartition(".")
        # A directive names a module or an attribute of one. An import error
        # other than a missing module is a bug in jbubble, so it propagates.
        try:
            importlib.import_module(path)
            continue
        except ImportError as error:
            if missing_optional(error):
                continue
            if not isinstance(error, ModuleNotFoundError):
                raise
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            if missing_optional(error):
                continue
            if not isinstance(error, ModuleNotFoundError):
                raise
            stale.append(path)
            continue
        if not hasattr(module, attr):
            stale.append(path)
    assert not stale, "Fix or remove these docs/api/ directives:\n" + "\n".join(stale)
