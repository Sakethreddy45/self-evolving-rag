import asyncio

from langchain_core.documents import Document

from selfrag.graph.schemas import Plan, Relevance
from selfrag.graph.state import GraphState
from selfrag.models import chat
from selfrag.prompts import ANSWER, GRADE, PLAN, REPLAN
from selfrag.retrieval.base import Retriever
from selfrag.settings import Role, settings

# provenance: which sub-query pulled this doc, so grading can score it
# against that query instead of the composite question
RETRIEVED_BY = "_retrieved_by"


async def plan(state: GraphState) -> GraphState:
    llm = chat(Role.REWRITE).with_structured_output(Plan)
    n = state.get("rewrites", 0)

    if n == 0:
        out = await llm.ainvoke(PLAN.format(question=state["question"]))
    else:
        out = await llm.ainvoke(
            REPLAN.format(queries=state["queries"], question=state["question"])
        )

    return {
        "queries": out.queries,
        "rewrites": n + 1,
        "steps": [{"node": "plan", "attempt": n + 1, "queries": out.queries}],
    }


def make_retrieve(retriever: Retriever):
    async def retrieve(state: GraphState) -> GraphState:
        queries = state["queries"]
        k = settings().retrieval.k

        results = await asyncio.gather(*(retriever.aretrieve(q, k=k) for q in queries))

        seen: dict[str, Document] = {}
        for query, res in zip(queries, results):
            for d in res.documents:
                key = d.id or d.page_content[:64]
                if key not in seen:
                    d.metadata[RETRIEVED_BY] = query
                    seen[key] = d

        return {
            "documents": list(seen.values()),
            "steps": [
                {
                    "node": "retrieve",
                    "strategy": results[0].strategy,
                    "n": len(seen),
                    "latency_ms": round(max(r.latency_ms for r in results)),
                }
            ],
        }

    return retrieve


async def grade(state: GraphState) -> GraphState:
    docs = state.get("documents", [])
    if not docs:
        return {"documents": [], "graded": [], "steps": [{"node": "grade", "kept": 0, "of": 0}]}

    llm = chat(Role.GRADE).with_structured_output(Relevance)
    sem = asyncio.Semaphore(settings().models[Role.GRADE].max_concurrency)

    async def one(doc: Document) -> Relevance:
        async with sem:
            return await llm.ainvoke(
                GRADE.format(
                    document=doc.page_content,
                    metadata={k: doc.metadata.get(k) for k in ("source", "published")},
                    query=doc.metadata[RETRIEVED_BY],
                )
            )

    verdicts = await asyncio.gather(*(one(d) for d in docs))
    kept = [d for d, v in zip(docs, verdicts) if v.on_topic and v.satisfies_constraints]

    return {
        "documents": kept,
        "graded": [
            {
                "source": d.metadata.get("source"),
                "query": d.metadata[RETRIEVED_BY],
                "on_topic": v.on_topic,
                "ok_constraints": v.satisfies_constraints,
                "reason": v.reason,
            }
            for d, v in zip(docs, verdicts)
        ],
        "steps": [{"node": "grade", "kept": len(kept), "of": len(docs)}],
    }


async def generate(state: GraphState) -> GraphState:
    docs = state.get("documents", [])
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    msg = await chat(Role.GENERATE).ainvoke(
        ANSWER.format(context=context, question=state["question"])
    )
    return {"answer": msg.content, "steps": [{"node": "generate", "n_docs": len(docs)}]}


async def give_up(state: GraphState) -> GraphState:
    return {
        "answer": "I couldn't find enough relevant material in the corpus to answer that.",
        "steps": [{"node": "give_up", "rewrites": state.get("rewrites", 0)}],
    }