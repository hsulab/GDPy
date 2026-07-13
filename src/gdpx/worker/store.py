import pathlib
from typing import Optional

from tinydb import Query, TinyDB


class JobRecord:
    """A single job record from the job store."""

    __slots__ = ("doc_id", "uid", "md5", "gdir", "group_number", "wdir_names")

    def __init__(self, doc_id: int, uid: str, md5: str, gdir: str, group_number: int, wdir_names: list[str]):
        self.doc_id = doc_id
        self.uid = uid
        self.md5 = md5
        self.gdir = gdir
        self.group_number = group_number
        self.wdir_names = wdir_names


class JobStore:
    """Persistence layer for job state tracking.

    Wraps TinyDB behind a stable interface so the rest of the worker
    code never touches TinyDB directly.  This also makes it possible
    to swap the backend later (e.g. SQLite) or mock it in tests.
    """

    def __init__(self, db_path: pathlib.Path):
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _search(self, *conditions) -> list[JobRecord]:
        try:
            with TinyDB(self._db_path, indent=2) as db:
                docs = db.search(conditions[0]) if conditions else db.all()
        except Exception:
            return []
        return [
            JobRecord(
                doc_id=d.doc_id,
                uid=d.get("uid", ""),
                md5=d.get("md5", ""),
                gdir=d.get("gdir", ""),
                group_number=d.get("group_number", -1),
                wdir_names=d.get("wdir_names", []),
            )
            for d in docs
        ]

    def get_running(self) -> list[JobRecord]:
        Q = Query()
        return self._search(Q.queued.exists() & (~Q.finished.exists()))

    def get_finished(self) -> list[JobRecord]:
        Q = Query()
        return self._search(Q.queued.exists() & Q.finished.exists())

    def get_retrieved(self) -> list[JobRecord]:
        Q = Query()
        return self._search(Q.queued.exists() & Q.finished.exists() & Q.retrieved.exists())

    def get_unretrieved(self) -> list[JobRecord]:
        Q = Query()
        return self._search(Q.queued.exists() & Q.finished.exists() & (~Q.retrieved.exists()))

    def get_by_gdir(self, gdir: str) -> Optional[JobRecord]:
        Q = Query()
        results = self._search(Q.gdir == gdir)
        return results[0] if results else None

    def get_queued(self) -> list[JobRecord]:
        Q = Query()
        return self._search(Q.queued.exists())

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def insert(self, uid: str, md5: str, gdir: str, group_number: int, wdir_names: list[str]) -> None:
        with TinyDB(self._db_path, indent=2) as db:
            db.insert(
                dict(
                    uid=uid,
                    md5=md5,
                    gdir=gdir,
                    group_number=group_number,
                    wdir_names=wdir_names,
                    queued=True,
                )
            )

    def mark_finished(self, gdir: str) -> None:
        with TinyDB(self._db_path, indent=2) as db:
            docs = db.search(Query().gdir == gdir)
            if docs:
                db.update({"finished": True}, doc_ids=[docs[0].doc_id])

    def mark_retrieved(self, gdir: str) -> None:
        with TinyDB(self._db_path, indent=2) as db:
            docs = db.search(Query().gdir == gdir)
            if docs:
                db.update({"retrieved": True}, doc_ids=[docs[0].doc_id])

    def remove_where(self, test_func) -> None:
        """Remove docs whose ``gdir`` satisfies *test_func(gdir) -> bool*."""
        with TinyDB(self._db_path, indent=2) as db:
            docs = db.all()
            ids = [d.doc_id for d in docs if test_func(d.get("gdir", ""))]
            if ids:
                db.remove(doc_ids=ids)

    @property
    def path(self) -> pathlib.Path:
        return self._db_path

    def __len__(self) -> int:
        try:
            with TinyDB(self._db_path, indent=2) as db:
                return len(db)
        except Exception:
            return 0
