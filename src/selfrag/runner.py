import time
from typing import Any

from selfrag.graph.build import build
from selfrag.retrieval.base import Retriever
from selfrag.settings import settings
from selfrag.store.trajectories import Trajectory, TrajectoryStore


def _outcome(state: dict[str, Any]) -> str:
    if any(s["node"] == "give_up" for s in state["steps"]):
        return "gave_up"
    v = next((s for s in reversed(state["steps"]) if s["node"] == "verify"), None)
    if v is None:
        return "unverified"
    if not v["grounded"]:
        return "ungrounded"
    if not v["useful"]:
        return "not_useful"
    return "answered"


class Runner:
    """Wraps the graph so every invocation is recorded. The graph itself
    stays unaware of logging."""

    def __init__(self, retriever: Retriever, store: TrajectoryStore | None = None):
        self._graph = build(retriever)
        self._store = store or TrajectoryStore(settings().trajectory_db)

    async def ask(self, question: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        state = await self._graph.ainvoke({"question": question})
        elapsed = round((time.perf_counter() - t0) * 1000)

        v = next((s for s in reversed(state["steps"]) if s["node"] == "verify"), None)
        self._store.add(
            Trajectory(
                question=question,
                outcome=_outcome(state),
                answer=state.get("answer"),
                steps=state["steps"],
                graded=state.get("graded", []),
                rewrites=state.get("rewrites", 0),
                regens=state.get("regens", 0),
                grounded=v["grounded"] if v else None,
                useful=v["useful"] if v else None,
                latency_ms=elapsed,
            )
        )
        return state