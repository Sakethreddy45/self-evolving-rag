import asyncio
import statistics
from collections import defaultdict

from selfrag.evals.dataset import Case, load
from selfrag.evals.score import score
from selfrag.learning.artifacts import Artifact, ArtifactStore
from selfrag.learning.diagnose import diagnose
from selfrag.retrieval.vector import VectorRetriever, open_store
from selfrag.runner import Runner
from selfrag.settings import settings

TRIALS = 5
MIN_GAIN = 0.15


async def measure(runner: Runner, cases: list[Case]) -> tuple[float, list[dict]]:
    recalls, traces = [], []
    for case in cases:
        per_case = []
        for _ in range(TRIALS):
            state = await runner.ask(case.question)
            s = await score(case, state.get("answer") or "")
            per_case.append(s.recall)
            traces.append({"steps": state["steps"], "missing": s.missing})
        recalls.append(statistics.mean(per_case))
    return statistics.mean(recalls), traces


async def main() -> None:
    cases = load()
    runner = Runner(VectorRetriever(store=open_store()))
    store = ArtifactStore(settings().artifact_db)

    by_tag: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        for t in c.tags:
            by_tag[t].append(c)

    scored = [(tag, await measure(runner, cs)) for tag, cs in by_tag.items()]
    worst_tag, (baseline, traces) = min(scored, key=lambda kv: kv[1][0])

    print(f"worst cluster: {worst_tag} at {baseline:.0%}")
    attempts = store.attempts_on(worst_tag)
    print(f"prior attempts on this cluster: {attempts}")

    d = await diagnose(by_tag[worst_tag], traces[: len(by_tag[worst_tag]) * TRIALS])
    print(f"\ncause: {d.cause}")
    print(f"fixable by prompt: {d.fixable_by_prompt}")

    if not d.fixable_by_prompt:
        print("\nescalate: this needs a new retrieval capability")
        return

    art = Artifact(
        kind="plan_hint",
        cluster=worst_tag,
        content=d.instruction,
        baseline=baseline,
        attempts=attempts + 1,
    )
    store.add(art)
    print(f"\nproposed hint:\n{d.instruction}\n")

    store.set_status(art.id, "live")
    after, _ = await measure(runner, by_tag[worst_tag])
    gain = after - baseline
    print(f"after: {after:.0%}  (gain {gain:+.0%})")

    if gain >= MIN_GAIN:
        store.set_status(art.id, "live", after)
        print("kept")
    else:
        store.set_status(art.id, "rejected", after)
        print("rejected — no measurable gain")
        if attempts + 1 >= 2:
            print(f"\nescalate: {worst_tag} has survived {attempts + 1} attempts")


if __name__ == "__main__":
    asyncio.run(main())