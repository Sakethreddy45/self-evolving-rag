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
    graded: list[dict[str, Any]]

    # append-only, so every node's contribution survives the merge
    steps: Annotated[list[dict[str, Any]], add]