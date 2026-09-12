from typing import Iterator

from selfrag.retrieval.base import Retriever


class StrategyRegistry:
    def __init__(self) -> None:
        self._live: dict[str, Retriever] = {}
        self._shadow: dict[str, Retriever] = {}

    def register(self, r: Retriever, *, shadow: bool = False) -> None:
        if not isinstance(r, Retriever):
            raise TypeError(f"{r!r} does not satisfy Retriever")
        (self._shadow if shadow else self._live)[r.name] = r

    def quarantine(self, name: str) -> None:
        self._live.pop(name, None)

    def get(self, name: str) -> Retriever:
        return self._live[name]

    def names(self) -> list[str]:
        return list(self._live)

    def shadows(self) -> Iterator[Retriever]:
        return iter(self._shadow.values())