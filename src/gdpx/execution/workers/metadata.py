"""Compact worker metadata with locked, atomic catalog updates.

The controller owns lifecycle records. Executing jobs append their scientific
results to job-specific journals, while legacy embedded result catalogs remain
readable and can still be merged explicitly by job UUID.
"""
import copy
import fcntl
import pathlib
import uuid
from collections.abc import Mapping
from contextlib import contextmanager

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.jsonio import decode, encode

from gdpx.execution.fingerprint import (
    atomic_write_text,
    payload_digest,
    read_structure_inputs,
    structure_digest,
    write_structure_inputs,
)
from .store import JobRecord, JobStore


class Catalog:
    def __init__(self, path, kind):
        self.path = pathlib.Path(path)
        self.kind = kind

    def empty(self):
        if self.kind == "inputs":
            return dict(format="gdpx-inputs", version=3, structures={}, workers={}, jobs={})
        return dict(format="gdpx-scheduler", version=1, providers={}, _default={}, results={})

    def read(self):
        if not self.path.exists():
            return self.empty()
        data = decode(self.path.read_text())
        required = ("structures", "workers", "jobs") if self.kind == "inputs" else ("providers", "_default", "results")
        if self.kind == "inputs" and data.get("format") == "gdpx-inputs" and data.get("version") == 1:
            raise ValueError("Legacy driver metadata has no frozen calculation set; use a new working directory.")
        expected_versions = (2, 3) if self.kind == "inputs" else (1,)
        if (data.get("format") != f"gdpx-{self.kind}" or data.get("version") not in expected_versions
                or any(not isinstance(data.get(key), dict) for key in required)):
            raise ValueError(f"Invalid {self.kind} catalog: {self.path}")
        if self.kind == "inputs":
            for worker in data["workers"]:
                WorkerMetadata.calculation_set(data, worker)
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
        if (self.root / "_data").exists() or any(self.root.glob("_*_jobs.json")):
            raise RuntimeError(f"Legacy driver worker layout at {self.root}; use a new working directory.")
        if self.inputs.path.exists() or self.state.path.exists():
            self.inputs.read()
            self.state.read()
            if not self.inputs.path.exists():
                raise ValueError("Missing inputs.json for compact driver metadata.")
            return True
        legacy = ("_scheduler.json", "compute-plan.json", "task_plan.json")
        if (any((self.directory / name).exists() for name in legacy)
                    or any(self.directory.glob("*.atoms.json"))
                    or any(self.directory.glob("job-*.json"))):
            raise ValueError("Legacy driver metadata has no frozen calculation set; use a new working directory.")
        return True

    def ensure(self):
        for catalog in (self.inputs, self.state):
            if not catalog.path.exists():
                with catalog.transaction():
                    pass

    def freeze_tasks(self, batches, machine_prefix):
        """Publish non-structural task inputs using the shared job catalogs."""
        definition = dict(version=1, batches=copy.deepcopy(batches), machine_prefix=machine_prefix)
        with self.inputs.transaction() as data:
            previous = data["workers"].get(self.worker)
            if previous is not None:
                if payload_digest(definition) != payload_digest(self.calculation_set(data, self.worker)):
                    raise ValueError("Calculation set conflict: training inputs changed. Use a new working directory.")
            else:
                data["workers"][self.worker] = dict(
                    calculation_set=definition, calculation_digest=payload_digest(definition),
                )
                for payload in batches:
                    uid = str(uuid.uuid4())
                    data["jobs"][uid] = dict(worker=self.worker, input=copy.deepcopy(payload),
                        job_digest=payload_digest(payload), machine_prefix=machine_prefix)
        return batches

    @staticmethod
    def structure_relative_path(digest):
        return pathlib.PurePosixPath(f"structures-{digest}.json")

    def structure_path(self, digest):
        relative = self.structure_relative_path(digest)
        return self.directory.joinpath(*relative.parts)

    @staticmethod
    def result_relative_path(uid):
        return pathlib.PurePosixPath(f"results-{uid}.jsonl")

    def result_path(self, uid):
        relative = self.result_relative_path(uid)
        return self.directory.joinpath(*relative.parts)

    def freeze_calculations(self, requests, *, complete=False, plan=None):
        """Compare or publish whole calculation sets in one atomic transaction.

        Requests are prepared in memory. No structure, manifest, or plan is
        published if any worker conflicts. Returned payloads contain saved seeds.
        """
        self.compact
        resolved = {}
        created_snapshots = []
        try:
            with self.inputs.transaction() as data:
                names = [request["worker"] for request in requests]
                if len(set(names)) != len(names):
                    raise ValueError("Duplicate workers in calculation set.")
                existing = set(data["workers"])
                if existing and (not set(names).issubset(existing) or (complete and set(names) != existing)):
                    raise ValueError("Calculation set conflict: workers changed. Use a new working directory.")
                for request in requests:
                    name = request["worker"]
                    definition = dict(version=1, batches=copy.deepcopy(request["batches"]),
                                      machine_prefix=request["machine_prefix"])
                    previous = data["workers"].get(name)
                    if previous is not None:
                        saved = self.calculation_set(data, name)
                        if request["reuse_saved_seeds"] and len(definition["batches"]) == len(saved["batches"]):
                            for batch, old in zip(definition["batches"], saved["batches"]):
                                batch["random_seeds"] = old["random_seeds"]
                        if payload_digest(definition) != payload_digest(saved):
                            changed = [key for key in ("structure_digest", "runtime", "indices", "structure_indices",
                                       "wdir_names", "driver_indices", "random_seeds", "share_random_seed")
                                       if [b.get(key) for b in definition["batches"]] != [b.get(key) for b in saved["batches"]]]
                            category = ", ".join(changed) or "batch mapping or machine prefix"
                            raise ValueError(f"Calculation set conflict: {category} changed. Use a new working directory.")
                    else:
                        frames = [frame.copy() for frame in request["frames"]]
                        for frame in frames:
                            frame.info = {}
                        digest = structure_digest(frames)
                        if data["version"] == 2:
                            if structure_digest(decode(encode(frames))) != digest:
                                raise ValueError("Structure input serialization changed its fingerprint.")
                            data["structures"].setdefault(digest, frames)
                        else:
                            relative = self.structure_relative_path(digest)
                            snapshot_path = self.directory.joinpath(*relative.parts)
                            if snapshot_path.exists():
                                read_structure_inputs(snapshot_path, digest)
                            else:
                                write_structure_inputs(snapshot_path, frames)
                                created_snapshots.append(snapshot_path)
                            data["structures"].setdefault(digest, {"file": relative.as_posix()})
                        data["workers"][name] = dict(
                            calculation_set=definition, calculation_digest=payload_digest(definition),
                            provenance={digest: dict(rows=request["provenance"], info=request["retained"])})
                        for payload in definition["batches"]:
                            uid = str(uuid.uuid4())
                            data["jobs"][uid] = dict(worker=name, input=payload, job_digest=payload_digest(payload),
                                                     machine_prefix=definition["machine_prefix"])
                    resolved[name] = definition["batches"]
                if plan is not None:
                    if "plan" in data and data["plan"] != plan:
                        raise ValueError("An immutable compute plan already exists. Use a new working directory.")
                    data["plan"] = plan
        except BaseException:
            for snapshot_path in created_snapshots:
                snapshot_path.unlink(missing_ok=True)
            raise
        return resolved

    @staticmethod
    def calculation_set(data, worker):
        record = data["workers"].get(worker, {})
        definition = record.get("calculation_set")
        if definition is None or definition.get("version") != 1:
            raise ValueError("Missing frozen calculation set; use a new working directory.")
        if payload_digest(definition) != record.get("calculation_digest"):
            raise ValueError("Calculation set fingerprint mismatch.")
        return definition

    def validate_payload(self, payload, machine_prefix):
        definition = self.calculation_set(self.inputs.read(), self.worker)
        if (machine_prefix != definition["machine_prefix"] or
                payload_digest(payload) not in {payload_digest(batch) for batch in definition["batches"]}):
            raise ValueError("Job is not in the frozen calculation set. Use a new working directory.")

    def frames(self, digest):
        data = self.inputs.read()
        snapshot = data["structures"][digest]
        if data["version"] == 3:
            if not isinstance(snapshot, dict) or set(snapshot) != {"file"}:
                raise ValueError("Invalid structure snapshot reference.")
            relative = pathlib.PurePosixPath(snapshot["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Structure snapshot path must stay inside the metadata directory.")
            return read_structure_inputs(self.directory.joinpath(*relative.parts), digest)
        frames = snapshot
        if structure_digest(frames) != digest:
            raise ValueError("Structure fingerprint mismatch.")
        return frames

    def provenance(self):
        return self.inputs.read()["workers"].get(self.worker, {}).get("provenance", {})

    def prepare_job(self, payload, machine_prefix):
        self.validate_payload(payload, machine_prefix)
        digest = payload_digest(payload)
        for uid, saved in self.inputs.read()["jobs"].items():
            if saved["worker"] == self.worker and saved["job_digest"] == digest:
                self.manifest(uid)
                return uid
        raise ValueError("Missing manifest for frozen calculation.")

    @staticmethod
    def validate_manifest(saved):
        if saved["input"].get("version") != 1 or payload_digest(saved["input"]) != saved["job_digest"]:
            raise ValueError("Job fingerprint mismatch.")
        return saved

    def manifest(self, uid):
        saved = self.validate_manifest(self.inputs.read()["jobs"][str(uid)])
        if saved["worker"] != self.worker:
            raise ValueError("Job belongs to a different worker.")
        self.validate_payload(saved["input"], saved["machine_prefix"])
        return saved

    @staticmethod
    def _validate_result_record(record, source):
        if not isinstance(record, Mapping) or set(record) != {"atoms", "results"}:
            raise ValueError(f"Invalid result record: {source}")
        frame = record["atoms"]
        name = frame.info.get("wdir")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Result record has no workdir: {source}")
        if not isinstance(record["results"], Mapping):
            raise ValueError(f"Invalid calculator results: {source}")
        return name

    @classmethod
    def _decode_result_lines(cls, content, source):
        content = content.encode() if isinstance(content, str) else content
        records = {}
        lines = content.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.endswith(b"\n"):
                if index == len(lines) - 1:
                    break
                raise ValueError(f"Incomplete result record: {source}")
            try:
                record = decode(line.decode().rstrip("\r\n"))
            except Exception as error:
                raise ValueError(f"Invalid result record: {source}") from error
            name = cls._validate_result_record(record, source)
            if name in records:
                raise ValueError(f"Duplicate result for {name}: {source}")
            records[name] = record
        return records

    def _external_result_records(self, uid):
        path = self.result_path(uid)
        if not path.exists():
            return {}
        return self._decode_result_lines(path.read_bytes(), path)

    def _result_records(self, uid):
        records = dict(self.state.read()["results"].get(str(uid), {}))
        for name, record in self._external_result_records(uid).items():
            previous = records.get(name)
            if previous is not None and encode(previous) != encode(record):
                raise ValueError(f"Conflicting result for {name} in job {uid}.")
            records[name] = record
        return records

    def results(self, uid):
        records = self._result_records(uid)
        frames = []
        for record in records.values():
            self._validate_result_record(record, self.result_path(uid))
            frame = record["atoms"]
            if record["results"]:
                frame.calc = SinglePointCalculator(frame, **record["results"])
            frames.append(frame)
        return frames

    @staticmethod
    def _truncate_incomplete_result(path):
        with open(path, "rb+") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            if end == 0:
                return
            handle.seek(-1, 2)
            if handle.read(1) == b"\n":
                return
            position = end
            while position:
                size = min(position, 8192)
                position -= size
                handle.seek(position)
                block = handle.read(size)
                offset = block.rfind(b"\n")
                if offset >= 0:
                    handle.truncate(position + offset + 1)
                    return
            handle.truncate(0)

    def put_result(self, uid, frame):
        saved = self.manifest(uid)
        name = frame.info["wdir"]
        if name not in saved["input"]["wdir_names"]:
            raise ValueError("Unexpected workdir in job results.")
        record = dict(atoms=frame.copy(), results=copy.deepcopy(getattr(frame.calc, "results", {})))
        content = encode(record)
        restored = decode(content)
        if self._validate_result_record(restored, self.result_path(uid)) != name:
            raise ValueError("Result serialization changed its workdir.")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.result_path(uid)
        with open(self.directory / ".metadata.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if path.exists():
                    self._truncate_incomplete_result(path)
                with open(path, "ab") as handle:
                    handle.write(content.encode() + b"\n")
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def merge_results(self, uid, remote):
        """Merge results embedded by an older remote worker."""
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
                raise ValueError("Scheduler provider changed for an existing worker.")
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
        removed = []
        with self.metadata.state.transaction() as data:
            for key, row in self._documents(data):
                if test_func(row["gdir"]):
                    removed.append(row["uid"])
                    del data["_default"][key]
                    data["results"].pop(row["uid"], None)
        for uid in removed:
            self.metadata.result_path(uid).unlink(missing_ok=True)

    def __len__(self):
        return len(self._documents(self.metadata.state.read()))
