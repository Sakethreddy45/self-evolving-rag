import asyncio
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from selfrag.evals.dataset import load
from selfrag.evals.score import score
from selfrag.evolution.gate import run_gate
from selfrag.evolution.generate import write_module
from selfrag.evolution.spec import write_spec
from selfrag.retrieval.vector import VectorRetriever, open_store
from selfrag.runner import Runner

MAX_ATTEMPTS = 3
WORKDIR = Path("data/generated")
BASELINE_CACHE = Path("data/baseline.json")

CLUSTERS = {
    "date-filter": (
        "retrieval is pure vector similarity and cannot filter on the publication "
        "date metadata that every chunk carries, so date-constrained questions "
        "return chunks from every year"
    ),
    "enumerate": (
        "these questions ask which posts exist within a date range, but retrieval "
        "returns the top k chunks ranked by similarity, which may all come from a "
        "single post and can never guarantee coverage of every distinct source"
    ),
}


def out_of_credits(detail: str) -> bool:
    return "insufficient_quota" in detail or "credit_balance" in detail


def corpus_metadata_keys() -> set[str]:
    sample = open_store().get(limit=1, include=["metadatas"])
    metas = sample.get("metadatas") or []
    return set(metas[0]) if metas else set()


async def _measure(cases: list) -> float:
    runner = Runner(VectorRetriever(store=open_store()))
    out = []
    for case in cases:
        rs = []
        for _ in range(3):
            state = await runner.ask(case.question)
            s = await score(case, state.get("answer") or "")
            rs.append(s.recall)
        out.append(statistics.mean(rs))
    return statistics.mean(out) if out else 1.0


async def baselines(cluster: str, target: list, others: list) -> tuple[float, float]:
    """Cached — the baseline doesn't change between attempts, and re-measuring
    it every run is the single most expensive thing here."""
    cache = json.loads(BASELINE_CACHE.read_text()) if BASELINE_CACHE.exists() else {}
    if cluster in cache:
        t, o = cache[cluster]
        print(f"  {cluster}: {t:.0%}   others: {o:.0%}  (cached)")
        return t, o

    t = await _measure(target)
    o = await _measure(others)
    cache[cluster] = [t, o]
    BASELINE_CACHE.write_text(json.dumps(cache))
    print(f"  {cluster}: {t:.0%}   others: {o:.0%}")
    return t, o


async def main(cluster: str) -> None:
    if cluster not in CLUSTERS:
        print(f"unknown cluster {cluster!r}; known: {', '.join(CLUSTERS)}")
        return

    WORKDIR.mkdir(parents=True, exist_ok=True)
    cases = load()

    by_tag = defaultdict(list)
    for c in cases:
        for t in c.tags:
            by_tag[t].append(c)

    target = by_tag[cluster]
    others = [c for c in cases if cluster not in c.tags]

    keys = corpus_metadata_keys()
    print(f"metadata fields: {sorted(keys)}")

    print("baseline")
    b_target, b_other = await baselines(cluster, target, others)
    print()

    spec = await write_spec(
        cluster, CLUSTERS[cluster], b_target, [c.question for c in target]
    )
    print(f"spec: {spec.name}")
    print(f"  {spec.summary}\n")

    violations = None
    previous = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"attempt {attempt}")
        source = await write_module(spec, violations, previous)
        (WORKDIR / f"{spec.name}_v{attempt}.py").write_text(source)

        result = await run_gate(
            source, spec.name, target, others, b_target, b_other, WORKDIR,
            metadata_keys=keys,
        )
        print(f"  {result.stage}: {result.detail}")
        if result.target_recall is not None:
            print(f"  {cluster}: {result.target_recall:.0%}  others: {result.other_recall:.0%}")

        if result.passed:
            print(f"\npromoted: {WORKDIR / f'{spec.name}_v{attempt}.py'}")
            return

        if out_of_credits(result.detail):
            print("\nout of credits — stopping")
            return

        violations = [result.detail]
        previous = source

    print(f"\nno candidate passed in {MAX_ATTEMPTS} attempts")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "enumerate"))