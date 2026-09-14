from typing import Literal

from langgraph.graph import END, START, StateGraph

from selfrag.graph.nodes import generate, give_up, grade, make_retrieve, plan
from selfrag.graph.state import GraphState
from selfrag.retrieval.base import Retriever
from selfrag.settings import settings


def after_grade(state: GraphState) -> Literal["generate", "plan", "give_up"]:
    p = settings().retrieval
    if len(state.get("documents", [])) >= p.min_surviving_docs:
        return "generate"
    if state.get("rewrites", 0) > p.max_rewrites:
        return "give_up"
    return "plan"


def build(retriever: Retriever):
    g = StateGraph(GraphState)

    g.add_node("plan", plan)
    g.add_node("retrieve", make_retrieve(retriever))
    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("give_up", give_up)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade)
    g.add_edge("generate", END)
    g.add_edge("give_up", END)

    return g.compile()