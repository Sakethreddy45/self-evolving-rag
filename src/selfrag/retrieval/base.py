from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    documents: list[Document]
    strategy: str
    latency_ms: float
    params: dict[str, Any] = field(default_factory=dict)
    scores: list[float] | None = None
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class Retriever(Protocol):
    """Contract every retrieval strategy satisfies, including generated ones.

    The registry checks conformance at registration time, so this doubles as
    the promotion gate's structural test.
    """

    @property
    def name(self) -> str: ...

    async def aretrieve(self, query: str, *, k: int, **kw: Any) -> RetrievalResult: ...