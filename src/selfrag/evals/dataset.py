from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path("evals/questions.yaml")


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    question: str
    must_contain: list[str]
    tags: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    expect_sources: list[str] = field(default_factory=list)


def load(path: Path = DEFAULT_PATH) -> list[Case]:
    raw = yaml.safe_load(path.read_text())
    return [Case(**c) for c in raw]