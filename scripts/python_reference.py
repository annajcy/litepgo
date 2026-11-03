"""Generate the Python API reference from the active CMake build of litepgo."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import pkgutil

from pdoc import doc, render


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        import litepgo
        from litepgo import fem
    except ImportError as error:
        raise SystemExit(
            "Build the Python extension first: cmake --preset release && "
            "cmake --build --preset release --target litepgo_python"
        ) from error
    if not Path(litepgo.__file__).resolve().is_relative_to(root):
        raise SystemExit("litepgo was imported from another checkout; build this checkout or set PYTHONPATH.")

    modules = {"litepgo": doc.Module(litepgo)}
    for info in pkgutil.walk_packages(litepgo.__path__, "litepgo."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        modules[info.name] = doc.Module(importlib.import_module(info.name))

    # Package __all__ exports symbols, so supply public submodules for pdoc's navigation.
    for name, module in modules.items():
        module.submodules = [item for key, item in modules.items() if key.rpartition(".")[0] == name]

    # Template instances are runtime objects, so pdoc cannot discover their classes itself.
    for name in fem.__all__:
        template = getattr(fem, name)
        if not isinstance(template, fem.Template):
            continue
        for args, instance in template.instances.items():
            label = f"{name}[{', '.join(getattr(arg, '__name__', str(arg)) for arg in args)}]"
            kind = doc.Class if inspect.isclass(instance) else doc.Function
            modules["litepgo.fem"].members[label] = kind(
                "litepgo.fem", label, instance, (instance.__module__, instance.__qualname__)
            )

    assert isinstance(modules["litepgo.energy"].get("Energy.eval"), doc.Function), "Missing native methods"
    assert all(
        isinstance(modules["litepgo.torch"].get(name), doc.Function) for name in ("energy", "response")
    ), "Missing Torch adapters"
    assert any(
        name.startswith("VolumetricFemEnergy[") and isinstance(member, doc.Class)
        for name, member in modules["litepgo.fem"].members.items()
    ), "Missing FEM template instances"

    output = root / "docs/public/reference/python"
    output.mkdir(parents=True, exist_ok=True)
    for path in output.rglob("*.html"):
        path.unlink()
    render.configure(docformat="markdown")
    for name, module in modules.items():
        path = output / f"{name.replace('.', '/')}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render.html_module(module, modules), encoding="utf-8")
    (output / "index.html").write_text(render.html_index(modules), encoding="utf-8")
    (output / "search.js").write_text(render.search_index(modules), encoding="utf-8")
    print(f"Generated Python reference for {len(modules)} modules in {output}")


if __name__ == "__main__":
    main()
