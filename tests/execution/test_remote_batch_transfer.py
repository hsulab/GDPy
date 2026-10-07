import pathlib
import shutil
from types import SimpleNamespace

from ase import Atoms

from gdpx.execution.factory import create_worker


class LocalSftp:
    """Exercise the real transfer code against a filesystem-backed remote tree."""

    def __init__(self, root):
        self.root = root

    def path(self, remote):
        return self.root / pathlib.PurePosixPath(remote).relative_to("/")

    def stat(self, path):
        return self.path(path).stat()

    lstat = stat

    def mkdir(self, path):
        self.path(path).mkdir()

    def put(self, local, remote):
        shutil.copyfile(local, self.path(remote))

    def get(self, remote, local):
        shutil.copyfile(self.path(remote), local)

    def listdir_attr(self, remote):
        return [SimpleNamespace(filename=p.name, st_mode=p.stat().st_mode)
                for p in self.path(remote).iterdir()]

    def close(self):
        pass


def test_batch_upload_and_fetch_are_isolated(tmp_path, monkeypatch):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    remote.mkdir()
    worker = create_worker({
        "schema_version": 4,
        "potential": {"provider": "emt"},
        "executor": {"provider": "ase", "method": "spc", "parameters": {}},
        "scheduler": {
            "provider": "slurm", "parameters": {},
            "transport": {"provider": "ssh", "parameters": {
                "hostname": "cluster", "remote_wdir": "/jobs",
            }},
        },
        "dispatch": {"batch_size": 2, "share_workdir": False},
    }, directory=local)
    for index in range(4):
        candidate = local / f"cand{index}"
        candidate.mkdir(parents=True)
        (candidate / "restart.txt").write_text(f"restart {index}")
    transport = worker.scheduler
    sftp = LocalSftp(remote)

    def submit(**kwargs):
        local_root, remote_root, _ = transport._roots()
        transport._transfer(sftp, local_root, remote_root)
        return "123"

    monkeypatch.setattr(transport, "submit", submit)
    worker.run([Atoms("Cu", positions=[[index, 0, 0]]) for index in range(4)])
    jobs = worker.job_store.get_queued()
    assert len(jobs) == 2
    job = jobs[1]
    worker._prepare_scheduler_for_job(job)
    local_root, remote_root, _ = transport._roots()
    staged = sftp.path(remote_root)
    assert {p.name for p in staged.glob("cand*")} == {"cand2", "cand3"}
    assert (staged / "cand2/restart.txt").read_text() == "restart 2"
    assert (staged / "_meta/inputs.json").exists()
    assert len(list((staged / "_meta").glob("structures-*.json"))) == 1
    assert {p.name for p in (staged / "_meta/jobscripts").iterdir()} == {
        f"run-{job.uid}.script",
    }
    assert not (staged / "_meta/scheduler.json").exists()

    # An older, polluted remote folder must not overwrite another batch's results.
    (staged / "cand0").mkdir()
    (staged / "cand0/restart.txt").write_text("stale unrelated output")
    (staged / "cand2/new-output.txt").write_text("new result")
    (local / "cand2/obsolete.txt").write_text("obsolete")
    journal = worker.metadata.result_path(job.uid)
    (staged / "_meta" / journal.name).write_text("job-owned journal")
    (staged / "_meta/jobscripts/slurm-123.out").write_text("current job log")

    class Client:
        def open_sftp(self):
            return sftp

        def close(self):
            pass

    monkeypatch.setattr(transport, "_client", Client)
    transport.sync_paths.add(local / "never-created.txt")
    transport.sync(job.wdir_names)
    assert (local / "cand0/restart.txt").read_text() == "restart 0"
    assert (local / "cand2/new-output.txt").read_text() == "new result"
    assert not (local / "cand2/obsolete.txt").exists()
    assert journal.read_text() == "job-owned journal"
    assert (local / "_meta/jobscripts/slurm-123.out").read_text() == "current job log"

    # A resubmission stages the current journal, but never the sibling journal.
    sibling = worker.metadata.result_path(jobs[0].uid)
    sibling.write_text("other batch journal")
    transport._transfer(sftp, local_root, remote_root)
    assert (staged / "cand0/restart.txt").read_text() == "stale unrelated output"
    assert (staged / "_meta" / journal.name).exists()
    assert not (staged / "_meta" / sibling.name).exists()
