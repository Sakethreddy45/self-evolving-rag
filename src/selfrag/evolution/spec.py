from pydantic import BaseModel, Field

from selfrag.models import chat
from selfrag.settings import Role

SPEC = """A failure cluster has survived two attempts at fixing it by prompting.
The capability is missing, so it needs new retrieval code.

Cluster: {cluster}
Root cause: {cause}
Recall on these cases: {recall:.0%}

Questions in this cluster:
{questions}

Available document metadata, on every chunk:
  source        post slug, e.g. "reward-hacking"
  url           full url
  published     iso date, e.g. "2024-11-28"
  published_ts  integer date, e.g. 20241128
  year          integer, e.g. 2024
  start_index   character offset of the chunk within its post

The store is Chroma. Its where-filter supports $eq, $ne, $gt, $gte, $lt, $lte,
$and, $or over metadata fields. A field takes exactly one operator, so a range
must be written as:
  {{"$and": [{{"published_ts": {{"$gte": 20230101}}}},
             {{"published_ts": {{"$lte": 20231231}}}}]}}

Chroma also exposes store.get(where=..., include=["metadatas", "documents"])
which returns every matching chunk without ranking them by similarity.

Write a short spec for a retrieval strategy that closes this gap. Say what it
does, how it decides what to return, and why that addresses the questions
above rather than the general case."""


class Spec(BaseModel):
    name: str = Field(description="short snake_case identifier for the strategy")
    summary: str = Field(description="what it does, two or three sentences")
    approach: str = Field(description="how it works, concretely")


async def write_spec(cluster: str, cause: str, recall: float, questions: list[str]) -> Spec:
    llm = chat(Role.GENERATE).with_structured_output(Spec)
    return await llm.ainvoke(
        SPEC.format(
            cluster=cluster,
            cause=cause,
            recall=recall,
            questions="\n".join(f"  - {q}" for q in questions),
        )
    )