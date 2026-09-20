import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

SCHEMA = """
create table if not exists artifacts (
    id        text primary key,
    at        text not null,
    kind      text not null,
    cluster   text not null,
    content   text not null,
    status    text not null,
    baseline  real,
    result    real,
    attempts  integer not null default 1
);
create index if not exists ix_status on artifacts(status, kind);
"""


@dataclass(slots=True)
class Artifact:
    """A prompt fragment the learning loop wrote, aimed at one failure cluster."""

    kind: str  # plan_hint | grade_hint | answer_hint
    cluster: str
    content: str
    status: str = "proposed"  # proposed | live | rejected
    baseline: float | None = None
    result: float | None = None
    attempts: int = 1
    id: str = field(default_factory=lambda: uuid4().hex)
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ArtifactStore:
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

    def add(self, a: Artifact) -> None:
        d = asdict(a)
        with self._conn() as c:
            c.execute(
                f"insert into artifacts ({', '.join(d)}) "
                f"values ({', '.join(f':{k}' for k in d)})",
                d,
            )

    def set_status(self, id: str, status: str, result: float | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "update artifacts set status = ?, result = ? where id = ?", (status, result, id)
            )

    def live(self, kind: str) -> list[Artifact]:
        return self._query("select * from artifacts where status='live' and kind=?", (kind,))

    def attempts_on(self, cluster: str) -> int:
        with self._conn() as c:
            row = c.execute(
                "select count(*) n from artifacts where cluster = ?", (cluster,)
            ).fetchone()
        return row["n"]

    def _query(self, sql: str, args: tuple) -> list[Artifact]:
        with self._conn() as c:
            return [Artifact(**dict(r)) for r in c.execute(sql, args).fetchall()]

