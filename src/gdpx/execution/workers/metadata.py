"""Compact driver metadata with locked, atomic catalog updates.

The controller owns lifecycle records. Executing jobs may only add their own
result records; remote result catalogs are merged explicitly by job UUID.
"""
import copy
import fcntl
import pathlib
import uuid
from contextlib import contextmanager

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.jsonio import decode, encode

from gdpx.execution.fingerprint import atomic_write_text, payload_digest, structure_digest
from .store import JobRecord, JobStore


class Catalog:
    def __init__(self, path, kind):
        self.path = pathlib.Path(path)
        self.kind = kind

    def empty(self):
        if self.kind == "inputs":
            return dict(format="gdpx-inputs", version=1, structures={}, workers={}, jobs={})
        return dict(format="gdpx-scheduler", version=1, providers={}, _default={}, results={})

    def read(self):
        if not self.path.exists():
            return self.empty()
        data = decode(self.path.read_text())
        required = ("structures", "workers", "jobs") if self.kind == "inputs" else ("providers", "_default", "results")
        if (data.get("format") != f"gdpx-{self.kind}" or data.get("version") != 1
                or any(not isinstance(data.get(key), dict) for key in required)):
            raise ValueError(f"Invalid {self.kind} catalog: {self.path}")
        return data

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the lock inode stable across atomic catalog replacements.
        with open(self.path.parent / ".metadata.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = self.read()
                before = encode(data)
                yield data
                after = encode(data)
                if before != after or not self.path.exists():
                    atomic_write_text(self.path, after)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


class WorkerMetadata:
    def __init__(self, root, worker="."):
        self.root = pathlib.Path(root)
        self.worker = worker
        self.directory = self.root / "_meta"
        self.inputs = Catalog(self.directory / "inputs.json", "inputs")
        self.state = Catalog(self.directory / "scheduler.json", "scheduler")

    @property
    def compact(self):
        if self.inputs.path.exists() or self.state.path.exists():
            self.inputs.read()
            self.state.read()
            if not self.inputs.path.exists():
                raise ValueError("Missing inputs.json for compact driver metadata.")
            return True
        legacy = ("_scheduler.json", "compute-plan.json", "task_plan.json")
        return not (any((self.directory / name).exists() for name in legacy)
                    or any(self.directory.glob("*.atoms.json"))
                    or any(self.directory.glob("job-*.json")))

    def ensure(self):
        for catalog in (self.inputs, self.state):
            if not catalog.path.exists():
                with catalog.transaction():
                    pass

    def put_structures(self, frames, provenance=(), retained=()):
        frames = [frame.copy() for frame in frames]
        for frame in frames:
            frame.info = {}
        digest = structure_digest(frames)
        snapshot = decode(encode(frames))
        if structure_digest(snapshot) != digest:
            raise ValueError("Structure input serialization changed its fingerprint.")
        with self.inputs.transaction() as data:
            if digest in data["structures"]:
                if structure_digest(data["structures"][digest]) != digest:
                    raise ValueError("Structure fingerprint mismatch.")
            else:
                data["structures"][digest] = frames
            worker = data["workers"].setdefault(self.worker, {"provenance": {}})
            worker["provenance"].setdefault(digest, dict(rows=list(provenance), info=list(retained)))
        return digest

    def frames(self, digest):
        frames = self.inputs.read()["structures"][digest]
        if structure_digest(frames) != digest:
            raise ValueError("Structure fingerprint mismatch.")
        return frames

    def provenance(self):
        return self.inputs.read()["workers"].get(self.worker, {}).get("provenance", {})

    def prepare_job(self, payload, machine_prefix):
        digest = payload_digest(payload)
        with self.inputs.transaction() as data:
            for uid, saved in data["jobs"].items():
                if saved["worker"] == self.worker and saved["job_digest"] == digest:
                    self.validate_manifest(saved)
                    if saved["machine_prefix"] != machine_prefix:
                        raise ValueError("Machine prefix changed for an existing job.")
                    return uid
            uid = str(uuid.uuid4())
            data["jobs"][uid] = dict(worker=self.worker, input=payload, job_digest=digest,
                                     machine_prefix=machine_prefix)
        return uid

    @staticmethod
    def validate_manifest(saved):
        if saved["input"].get("version") != 1 or payload_digest(saved["input"]) != saved["job_digest"]:
            raise ValueError("Job fingerprint mismatch.")
        return saved

    def manifest(self, uid):
        saved = self.validate_manifest(self.inputs.read()["jobs"][str(uid)])
        if saved["worker"] != self.worker:
            raise ValueError("Job belongs to a different worker.")
        return saved

    def put_plan(self, plan):
        with self.inputs.transaction() as data:
            if "plan" in data and data["plan"] != plan:
                raise ValueError("An immutable compute plan already exists.")
            data["plan"] = plan

    def results(self, uid):
        records = self.state.read()["results"].get(str(uid), {})
        frames = []
        for record in records.values():
            frame = record["atoms"]
            if record["results"]:
                frame.calc = SinglePointCalculator(frame, **record["results"])
            frames.append(frame)
        return frames

    def put_result(self, uid, frame):
        saved = self.manifest(uid)
        name = frame.info["wdir"]
        if name not in saved["input"]["wdir_names"]:
            raise ValueError("Unexpected workdir in job results.")
        record = dict(atoms=frame.copy(), results=copy.deepcopy(getattr(frame.calc, "results", {})))
        with self.state.transaction() as data:
            data["results"].setdefault(str(uid), {})[name] = record

    def merge_results(self, uid, remote):
        records = remote.get("results", {}).get(str(uid), {})
        saved = self.manifest(uid)
        if not set(records).issubset(saved["input"]["wdir_names"]):
            raise ValueError("Remote results contain unexpected workdirs.")
        if any(record["atoms"].info.get("wdir") != name for name, record in records.items()):
            raise ValueError("Remote result workdir mismatch.")
        with self.state.transaction() as data:
            data["results"].setdefault(str(uid), {}).update(records)


class CatalogJobStore(JobStore):
    def __init__(self, metadata, provider):
        super().__init__(metadata.state.path)
        self.metadata = metadata
        with metadata.state.transaction() as data:
            previous = data["providers"].get(metadata.worker)
            if previous is not None and previous != provider:
                raise ValueError("Scheduler provider changed for an existing driver worker.")
            data["providers"][metadata.worker] = provider

    def _documents(self, data):
        return [(key, row) for key, row in data["_default"].items()
                if row["worker"] == self.metadata.worker]

    def _search(self, *conditions):
        return [JobRecord(doc_id=int(key), uid=row["uid"], md5="", gdir=row["gdir"],
                          group_number=row["group_number"], wdir_names=row["wdir_names"],
                          scheduler_job_id=row.get("scheduler_job_id", ""), attempt=row.get("attempt", 0),
                          structure_digest=row["structure_digest"], job_digest=row["job_digest"])
                for key, row in self._documents(self.metadata.state.read())
                if not conditions or conditions[0](row)]

    def insert(self, uid, md5, gdir, group_number, wdir_names, *, structure_digest="", job_digest=""):
        with self.metadata.state.transaction() as data:
            if any(row["gdir"] == gdir for _, row in self._documents(data)):
                return
            key = str(max(map(int, data["_default"]), default=0) + 1)
            data["_default"][key] = dict(worker=self.metadata.worker, uid=uid, gdir=gdir,
                group_number=group_number, wdir_names=wdir_names, structure_digest=structure_digest,
                job_digest=job_digest, queued=True, attempt=0)

    def _update(self, gdir, change):
        with self.metadata.state.transaction() as data:
            for _, row in self._documents(data):
                if row["gdir"] == gdir:
                    change(row)

    def mark_submitted(self, gdir, scheduler_job_id):
        self._update(gdir, lambda row: row.update(
            scheduler_job_id=str(scheduler_job_id), attempt=row["attempt"] + 1))

    def mark_finished(self, gdir):
        self._update(gdir, lambda row: row.update(finished=True))

    def mark_retrieved(self, gdir):
        self._update(gdir, lambda row: row.update(retrieved=True))

    def remove_where(self, test_func):
        with self.metadata.state.transaction() as data:
            for key, row in self._documents(data):
                if test_func(row["gdir"]):
                    del data["_default"][key]
                    data["results"].pop(row["uid"], None)

    def __len__(self):
        return len(self._documents(self.metadata.state.read()))
