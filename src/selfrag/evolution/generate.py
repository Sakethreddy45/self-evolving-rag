from pydantic import BaseModel, Field

from selfrag.evolution.load import class_name
from selfrag.evolution.spec import Spec
from selfrag.models import chat
from selfrag.settings import Role

WRITE = """Write a retrieval strategy as a single Python module.

Spec
  {summary}
  {approach}

Define a class named exactly {cls} satisfying this protocol:

    @property
    def name(self) -> str: ...
    async def aretrieve(self, query: str, *, k: int, **kw) -> RetrievalResult: ...

Constructor: __init__(self, store, *, name="{spec_name}")

RetrievalResult is a frozen dataclass. Import it, do not redefine it:

    from selfrag.retrieval.base import RetrievalResult

    RetrievalResult(documents=[...], strategy=str, latency_ms=float,
                    params=dict, scores=[float] | None)

The store is a langchain_chroma.Chroma with two retrieval methods:

    hits = await store.asimilarity_search_with_score(query, k=k, filter=<where|None>)
      returns list[tuple[Document, float]] — unpack as:
          docs = [d for d, _ in hits]
          scores = [s for _, s in hits]

    res = store.get(where=<where|None>, include=["metadatas", "documents"])
      synchronous, no ranking, returns every match as parallel lists:
          res["ids"]        list[str]
          res["documents"]  list[str]      the chunk text
          res["metadatas"]  list[dict]     one dict per chunk
      build Documents yourself:
          Document(page_content=t, metadata=m)
              for t, m in zip(res["documents"], res["metadatas"])

A Document has .page_content and .metadata attributes. It is not subscriptable.

Documents you return must carry real chunk text in page_content. A Document
whose page_content is only metadata — a slug, a date, an id — is useless to
the answer generator, which cannot describe a post it has not read. If you
use store.get(), include "documents" and use that text.

Chroma where-filters take exactly one operator per field, so a range is:
    {{"$and": [{{"published_ts": {{"$gte": 20230101}}}},
               {{"published_ts": {{"$lte": 20231231}}}}]}}

The query is free text, e.g. "what did she write about in 2024". Extract any
year with a regex — never call int() on the raw query. Handle queries with no
year at all by returning an unfiltered similarity search.

Imports restricted to: __future__, time, typing, dataclasses, datetime, re,
collections, langchain_core.documents, langchain_chroma, selfrag.retrieval.base,
selfrag.settings. No file, network, or subprocess access.

Return only module source. No markdown fences, no commentary.

{feedback}"""

RETRY = """Your previous attempt was rejected:

{violations}

Here is what you wrote:

{previous}

Fix that specific problem and return the corrected module."""


class Module(BaseModel):
    source: str = Field(description="complete python module source")


async def write_module(
    spec: Spec, violations: list[str] | None = None, previous: str | None = None
) -> str:
    feedback = ""
    if violations:
        feedback = RETRY.format(
            violations="\n".join(f"- {v}" for v in violations), previous=previous or ""
        )

    llm = chat(Role.GENERATE).with_structured_output(Module)
    out = await llm.ainvoke(
        WRITE.format(
            cls=class_name(spec.name),
            spec_name=spec.name,
            summary=spec.summary,
            approach=spec.approach,
            feedback=feedback,
        )
    )
    return out.source