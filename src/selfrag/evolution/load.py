import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module(source: str, name: str, workdir: Path) -> ModuleType:
    path = workdir / f"{name}.py"
    path.write_text(source)

    spec = importlib.util.spec_from_file_location(f"selfrag_generated.{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def class_name(spec_name: str) -> str:
    """date_filter_retrieval -> DateFilterRetrieval"""
    return "".join(p.title() for p in spec_name.split("_"))


def find_retriever(mod: ModuleType, expected: str) -> type:
    cls = getattr(mod, expected, None)
    if cls is None:
        found = [k for k, v in vars(mod).items() if isinstance(v, type)]
        raise TypeError(f"no class named {expected}; module defines {found}")
    if not isinstance(cls, type) or not hasattr(cls, "aretrieve"):
        raise TypeError(f"{expected} has no aretrieve method")
    return cls