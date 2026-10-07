from types import SimpleNamespace

import pytest

from gdpx.core.output import Box, quiet_logging
from gdpx.execution.workers.worker import BaseWorker


@pytest.mark.parametrize("resubmit", [False, True])
def test_remote_batch_progress_preserves_lifecycle(tmp_path, monkeypatch, resubmit):
    class Worker(BaseWorker):
        def run(self):
            pass

        def _sync_job(self, job):
            synced.append(job.gdir)

        def _check_job_convergence(self, job):
            return job.group_number == 0

        def _resubmit_job(self, job):
            self.job_store.mark_submitted(job.gdir, "new-id")

    worker = Worker(tmp_path)
    worker._scheduler = SimpleNamespace(
        name="slurm", transport_name="ssh", hostname="cluster",
        is_finished=lambda: worker.scheduler.job_name != "batch2",
    )
    for index in range(4):
        worker.job_store.insert(f"uid{index}", "digest", f"batch{index}", index, [f"cand{index}"])
        worker.job_store.mark_submitted(f"batch{index}", str(100 + index))
    worker.job_store.mark_finished("batch3")
    # Historical batches must not inflate the current batch denominator.
    worker.job_store.insert("old", "old-digest", "historical", 0, ["old"])
    worker.job_store.mark_finished("historical")
    ticks = iter([10.0, 12.0, 20.0, 23.0])
    monkeypatch.setattr("gdpx.execution.workers.worker.time.monotonic", lambda: next(ticks))
    synced = []
    lines = []

    with quiet_logging(), Box("test", emit=lines.append).as_parent():
        worker.inspect(resubmit=resubmit)

    output = "\n".join(lines)
    assert "batch 1/4 | job 100 | finished | 1 calculations" in output
    assert "batch 1/4 | fetching from cluster..." in output
    assert "batch 1/4 | fetched | elapsed 2.0 s" in output
    assert "batch 2/4 | fetched | elapsed 3.0 s" in output
    assert "batch 3/4 | job 102 | queued/running | 1 calculations" in output
    assert synced == ["batch0", "batch1"]
    assert {job.gdir for job in worker.job_store.get_running()} == {"batch1", "batch2"}
    if resubmit:
        assert "batch 2/4 | incomplete calculations | resubmitting..." in output
        assert "batch 2/4 | resubmitted | job new-id" in output
    else:
        assert "batch 2/4 | incomplete calculations | manual resubmission required" in output
