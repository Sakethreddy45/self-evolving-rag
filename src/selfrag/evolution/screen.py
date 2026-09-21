import ast
from dataclasses import dataclass

ALLOWED_ROOTS = {
    "__future__",
    "time",
    "typing",
    "dataclasses",
    "datetime",
    "re",
    "collections",
    "langchain_core",
    "langchain_chroma",
    "selfrag",
}

BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input"}

BANNED_ATTRS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "remove"),
    ("subprocess", "run"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("shutil", "rmtree"),
    ("sys", "exit"),
}

# contracts the module must import rather than define its own copy of
PROTECTED_NAMES = {"RetrievalResult", "Retriever"}


@dataclass(frozen=True, slots=True)
class ScreenResult:
    ok: bool
    violations: list[str]


def _imports(tree: ast.AST) -> list[tuple[str, str]]:
    """(root_package, full_name) for every import in the module."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name.split(".")[0], a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out.append((mod.split(".")[0], mod))
    return out


def screen(source: str, *, metadata_keys: set[str] | None = None) -> ScreenResult:
    """Static checks run before the module is executed. Everything here is
    deterministic and cheap — it exists so the expensive stages never see
    code that was never going to work."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ScreenResult(False, [f"syntax error: {e}"])

    bad: list[str] = []

    for root, full in _imports(tree):
        if root not in ALLOWED_ROOTS:
            bad.append(f"import not allowed: {full}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BANNED_CALLS:
                bad.append(f"banned call: {f.id}")
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if (f.value.id, f.attr) in BANNED_ATTRS:
                    bad.append(f"banned call: {f.value.id}.{f.attr}")

        elif isinstance(node, ast.ClassDef) and node.name in PROTECTED_NAMES:
            bad.append(
                f"redefines {node.name}; import it from selfrag.retrieval.base "
                f"instead of declaring a local copy"
            )

    # a filter referencing a field the corpus doesn't have matches nothing and
    # fails silently, so catch unknown metadata keys before anything runs
    if metadata_keys:
        referenced = _metadata_keys_used(tree)
        unknown = referenced - metadata_keys
        if unknown:
            bad.append(
                f"unknown metadata fields {sorted(unknown)}; "
                f"available fields are {sorted(metadata_keys)}"
            )

    return ScreenResult(not bad, bad)


def _metadata_keys_used(tree: ast.AST) -> set[str]:
    """String keys in dict literals that look like filter clauses — a key
    whose value is a dict of chroma operators, or which sits beside one."""
    ops = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
    found: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            name = key.value
            if name.startswith("$"):
                continue
            if isinstance(value, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value in ops for k in value.keys
            ):
                found.add(name)
    return found