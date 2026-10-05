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
    # TODO(v0.2.0): remove this mark once the API changes for the release have
    # merged, so that a missing or stale directive fails the suite.
    pytest.mark.xfail(
        strict=False,
        reason="the v0.2.0 API changes are still merging, and docs/api/ "
        "catches up with them before the release",
    ),
]
# Optional dependencies. A module that needs a missing one is skipped.
OPTIONAL = {"h5py", "matplotlib"}


def public_modules():
    names = ["jbubble"] + [
        info.name
        for info in pkgutil.walk_packages(jbubble.__path__, prefix="jbubble.")
        if not any(part.startswith("_") for part in info.name.split("."))
    ]
    for name in names:
        try:
            yield importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name not in OPTIONAL:
                raise


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
        try:
            importlib.import_module(path)
            continue
        except ModuleNotFoundError:
            pass
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            stale.append(path)
            continue
        if not hasattr(module, attr):
            stale.append(path)
    assert not stale, "Fix or remove these docs/api/ directives:\n" + "\n".join(stale)
