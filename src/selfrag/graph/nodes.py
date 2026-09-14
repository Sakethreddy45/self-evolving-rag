import asyncio

from langchain_core.documents import Document

from selfrag.graph.schemas import Grounding, Plan, Relevance, Usefulness
from selfrag.graph.state import GraphState
from selfrag.models import chat
from selfrag.prompts import ANSWER, GRADE, GROUND, PLAN, REPLAN, STRICTER, USEFUL
from selfrag.retrieval.base import Retriever
from selfrag.settings import Role, settings


RETRIEVED_BY = "_retrieved_by"

def _context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


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
                    document=doc.page_content[:600],
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
    unsupported = state.get("unsupported") or []

    prompt = (
        STRICTER.format(
            context=_context(docs),
            question=state["question"],
            unsupported="\n".join(f"- {c}" for c in unsupported),
        )
        if unsupported
        else ANSWER.format(context=_context(docs), question=state["question"])
    )

    msg = await chat(Role.GENERATE).ainvoke(prompt)
    n = state.get("regens", 0)
    return {
        "answer": msg.content,
        "regens": n + 1,
        "steps": [{"node": "generate", "n_docs": len(docs), "attempt": n + 1}],
    }

async def verify(state: GraphState) -> GraphState:
    context = _context(state.get("documents", []))
    answer = state["answer"]

    # independent checks, so run them together
    ground, useful = await asyncio.gather(
        chat(Role.GRADE)
        .with_structured_output(Grounding)
        .ainvoke(GROUND.format(context=context, answer=answer)),
        chat(Role.GRADE)
        .with_structured_output(Usefulness)
        .ainvoke(USEFUL.format(question=state["question"], context=context, answer=answer)),
    )

    return {
        "unsupported": ground.unsupported,
        "steps": [
            {
                "node": "verify",
                "grounded": ground.grounded,
                "useful": useful.useful,
                "unsupported": len(ground.unsupported),
                "reason": useful.reason,
            }
        ],
    }

async def give_up(state: GraphState) -> GraphState:
    return {
        "answer": "I couldn't find enough relevant material in the corpus to answer that.",
        "steps": [{"node": "give_up", "rewrites": state.get("rewrites", 0)}],
    }