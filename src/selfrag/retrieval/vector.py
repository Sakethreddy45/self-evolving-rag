import time
from typing import Any

from langchain_chroma import Chroma

from selfrag.models import embeddings
from selfrag.retrieval.base import RetrievalResult
from selfrag.settings import settings


class VectorRetriever:
    def __init__(self, store: Chroma, *, name: str = "vector"):
        self._store = store
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def aretrieve(self, query: str, *, k: int, **kw: Any) -> RetrievalResult:
        t0 = time.perf_counter()

        hits = await self._store.asimilarity_search_with_score(query, k=k, filter=kw.get("where"))
        return RetrievalResult(
            documents=[d for d, _ in hits],
            strategy=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000,
            params={"k": k, **kw},
            scores=[s for _, s in hits],
        )


def open_store() -> Chroma:
    s = settings()
    return Chroma(
        collection_name=s.collection,
        embedding_function=embeddings(),
        persist_directory=str(s.chroma_dir),
    )