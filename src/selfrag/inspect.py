import sys
from collections import Counter

from selfrag.settings import settings
from selfrag.store.trajectories import TrajectoryStore


def main() -> None:
    store = TrajectoryStore(settings().trajectory_db)
    recent = store.recent(limit=200)

    if not recent:
        print("no trajectories yet")
        return

    counts = Counter(t.outcome for t in recent)
    for outcome, n in counts.most_common():
        print(f"{n:4}  {outcome}")

    print()
    for t in store.failures(limit=int(sys.argv[1]) if len(sys.argv) > 1 else 10):
        print(f"{t.outcome:12} rw={t.rewrites} {t.latency_ms:6}ms  {t.question}")


if __name__ == "__main__":
    main()