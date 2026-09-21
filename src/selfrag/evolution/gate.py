import statistics
from dataclasses import dataclass
from pathlib import Path

from selfrag.evals.dataset import Case
from selfrag.evals.score import score
from selfrag.evolution.load import class_name, find_retriever, load_module
from selfrag.evolution.screen import screen
from selfrag.retrieval.base import RetrievalResult, Retriever
from selfrag.retrieval.vector import open_store
from selfrag.runner import Runner

TRIALS = 3
MIN_GAIN = 0.15
MAX_REGRESSION = 0.05
MAX_LATENCY_MS = 5000
MIN_CHUNK_CHARS = 200


@dataclass(slots=True)
class GateResult:
    stage: str
    passed: bool
    detail: str
    target_recall: float | None = None
    other_recall: float | None = None


async def _smoke(instance, probes: list[str]) -> str | None:
    """Cheap behavioural checks. A strategy can fail by raising or by being
    inert, and recall alone can't tell those apart."""
    for q in probes:
        try:
            res = await instance.aretrieve(q, k=5)
        except Exception as e:
            return f"aretrieve raised on {q!r}: {type(e).__name__}: {e}"

        if not isinstance(res, RetrievalResult):
            return f"returned {type(res).__name__}, not the imported RetrievalResult"
        if not res.documents:
            return f"returned zero documents for {q!r}"
        if res.scores is not None and len(res.scores) != len(res.documents):
            return f"scores/documents length mismatch: {len(res.scores)} vs {len(res.documents)}"
        if res.latency_ms > MAX_LATENCY_MS:
            return f"too slow: {res.latency_ms:.0f}ms on {q!r}"
    return None


async def _retrieval_check(instance, cases: list[Case]) -> str | None:
    """Zero LLM calls. Checks the retriever against known-correct sources, so
    a broken strategy is rejected in a second rather than after a full
    evaluation run."""
    for case in cases:
        if not case.expect_sources:
            continue

        try:
            res = await instance.aretrieve(case.question, k=20)
        except Exception as e:
            return f"{case.id}: aretrieve raised {type(e).__name__}: {e}"

        got = {d.metadata.get("source") for d in res.documents}
        want = set(case.expect_sources)

        if missing := want - got:
            return f"{case.id}: never retrieved {sorted(missing)}"

        # a document whose text is just metadata can't be described by the
        # answer model, so require real chunk text
        for d in res.documents:
            if d.metadata.get("source") in want and len(d.page_content) < MIN_CHUNK_CHARS:
                return (
                    f"{case.id}: page_content for {d.metadata.get('source')!r} is only "
                    f"{len(d.page_content)} chars — return the chunk text from "
                    f'include=["documents"], not a metadata summary'
                )
    return None


async def _recall(runner: Runner, cases: list[Case], trials: int) -> float:
    per_case = []
    for case in cases:
        rs = []
        for _ in range(trials):
            state = await runner.ask(case.question)
            s = await score(case, state.get("answer") or "")
            rs.append(s.recall)
        per_case.append(statistics.mean(rs))
    return statistics.mean(per_case) if per_case else 1.0


async def run_gate(
    source: str,
    name: str,
    target: list[Case],
    others: list[Case],
    baseline_target: float,
    baseline_other: float,
    workdir: Path,
    metadata_keys: set[str] | None = None,
) -> GateResult:
    s = screen(source, metadata_keys=metadata_keys)
    if not s.ok:
        return GateResult("screen", False, "; ".join(s.violations))

    try:
        mod = load_module(source, name, workdir)
        cls = find_retriever(mod, expected=class_name(name))
        instance = cls(open_store())
    except Exception as e:
        return GateResult("load", False, f"{type(e).__name__}: {e}")

    if not isinstance(instance, Retriever):
        return GateResult("conform", False, "does not satisfy the Retriever protocol")

    probes = [c.question for c in target[:2]] + [c.question for c in others[:1]]
    if err := await _smoke(instance, probes):
        return GateResult("smoke", False, err)

    # everything above is free; everything below costs api calls
    if err := await _retrieval_check(instance, target):
        return GateResult("retrieval", False, err)

    runner = Runner(instance)
    try:
        target_recall = await _recall(runner, target, TRIALS)
        other_recall = await _recall(runner, others, TRIALS)
    except Exception as e:
        return GateResult("evaluate", False, f"{type(e).__name__}: {e}")

    gain = target_recall - baseline_target
    regression = baseline_other - other_recall

    if gain < MIN_GAIN:
        return GateResult(
            "evaluate", False, f"gain {gain:+.0%} below {MIN_GAIN:.0%}",
            target_recall, other_recall,
        )
    if regression > MAX_REGRESSION:
        return GateResult(
            "evaluate", False, f"regressed unrelated cases by {regression:.0%}",
            target_recall, other_recall,
        )

    return GateResult(
        "promote", True, f"gain {gain:+.0%}, no regression", target_recall, other_recall
    )