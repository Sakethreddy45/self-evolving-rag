from operator import add
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """Flows through every node. Each node returns a partial update, and
    LangGraph merges it in."""

    question: str
    queries: list[str]
    documents: list[Document]
    answer: str
    rewrites: int
    regens: int
    graded: list[dict[str, Any]]
    unsupported: list[str]

    # set by retrieve when the strategy returned a complete set rather than
    # a ranked one, which changes what grading should do with it
    enumerated: bool

    # append-only, so every node's contribution survives the merge
    steps: Annotated[list[dict[str, Any]], add]