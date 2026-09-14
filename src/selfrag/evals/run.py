import asyncio
import statistics
import sys
from collections import Counter, defaultdict

from selfrag.evals.dataset import load
from selfrag.evals.score import score
from selfrag.retrieval.vector import VectorRetriever, open_store
from selfrag.runner import Runner


async def main(trials: int = 3) -> None:
    cases = load()
    runner = Runner(VectorRetriever(store=open_store()))
    by_tag: dict[str, list[float]] = defaultdict(list)

    for case in cases:
        recalls = []
        missed: Counter[str] = Counter()

        for _ in range(trials):
            state = await runner.ask(case.question)
            s = await score(case, state.get("answer") or "")
            recalls.append(s.recall)
            missed.update(s.missing)

        mean = statistics.mean(recalls)

        spread = "".join("+" if r == 1.0 else "." for r in recalls)
        print(f"{mean:5.0%}  {spread}  {case.id}")

        for fact, n in missed.most_common():
            label = "never found" if n == trials else f"missed {n}/{trials}"
            print(f"              {label}: {fact}")

        for tag in case.tags:
            by_tag[tag].append(mean)

    print("\nrecall by tag")
    for tag, vals in sorted(by_tag.items(), key=lambda kv: statistics.mean(kv[1])):
        print(f"  {statistics.mean(vals):5.0%}  {tag}  ({len(vals)} cases)")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))