import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

SCHEMA = """
create table if not exists trajectories (
    id          text primary key,
    at          text not null,
    question    text not null,
    outcome     text not null,
    answer      text,
    steps       text not null,
    graded      text not null,
    rewrites    integer not null,
    regens      integer not null,
    grounded    integer,
    useful      integer,
    latency_ms  integer not null
);
create index if not exists ix_outcome on trajectories(outcome);
create index if not exists ix_at on trajectories(at);
"""


@dataclass(slots=True)
class Trajectory:
    question: str
    outcome: str
    steps: list[dict[str, Any]]
    graded: list[dict[str, Any]]
    rewrites: int
    regens: int
    latency_ms: int
    answer: str | None = None
    grounded: bool | None = None
    useful: bool | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class TrajectoryStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def add(self, t: Trajectory) -> None:
        d = asdict(t)
        d["steps"] = json.dumps(d["steps"])
        d["graded"] = json.dumps(d["graded"])
        cols = ", ".join(d)
        marks = ", ".join(f":{k}" for k in d)
        with self._conn() as c:
            c.execute(f"insert into trajectories ({cols}) values ({marks})", d)

    def failures(self, limit: int = 100) -> list[Trajectory]:
        return self._query(
            "select * from trajectories where outcome != 'answered' "
            "order by at desc limit ?",
            (limit,),
        )

    def recent(self, limit: int = 20) -> list[Trajectory]:
        return self._query("select * from trajectories order by at desc limit ?", (limit,))

    def _query(self, sql: str, args: tuple) -> list[Trajectory]:
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [
            Trajectory(
                **{
                    **dict(r),
                    "steps": json.loads(r["steps"]),
                    "graded": json.loads(r["graded"]),
                    "grounded": None if r["grounded"] is None else bool(r["grounded"]),
                    "useful": None if r["useful"] is None else bool(r["useful"]),
                }
            )
            for r in rows
        ]