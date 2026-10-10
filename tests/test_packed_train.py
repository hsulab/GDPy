import os
import subprocess
import pytest
from gdpx.execution.schedulers.slurm import SlurmScheduler
from gdpx.execution.workers.train import TrainerBasedWorker
from gdpx.providers import ComponentConfig

class Dataset:
    def as_dict(self):
        return {"name": "xyz", "dataset_path": "dataset"}

class Trainer:
    random_seed = 42
    directory = None
    component_config = ComponentConfig("deepmd", "default", {})
    def set_rng(self, seed):
        pass
    def read_convergence(self):
        return self.directory.name in self.finished
    def __init__(self):
        self.finished = set()
    def freeze(self):
        path = self.directory / "deepmd.pt2"
        path.touch()
        return path

def make_worker(tmp_path, concurrent):
    trainer = Trainer()
    scheduler = SlurmScheduler(concurrent_tasks=concurrent, ntasks=concurrent)
    scheduler.machine_prefix = ""
    submitted = []
    scheduler.submit = lambda **kw: submitted.append(scheduler.script) or str(len(submitted))
    scheduler.is_finished = lambda: False
    worker = TrainerBasedWorker(trainer, scheduler, directory=tmp_path)
    return worker, submitted

def test_packed_grouping_and_lifecycle(tmp_path):
    worker, submitted = make_worker(tmp_path, 4)
    worker.run(Dataset(), size=5)
    assert len(submitted) == 2
    worker.run(Dataset(), size=5)
    assert len(submitted) == 2
    assert [r.wdir_names for r in worker.job_store.get_queued()] == [["m0", "m1", "m2", "m3"], ["m4"]]
    assert worker.job_store.path == tmp_path / "_meta/scheduler.json"
    assert submitted[0].parent == tmp_path / "_meta/jobscripts"
    assert not (tmp_path / "_slurm_jobs.json").exists()
    assert not (tmp_path / "m0/train.script").exists()
    worker.scheduler.is_finished = lambda: True
    worker.trainer.finished = {"m0", "m1", "m2", "m4"}
    worker.inspect(resubmit=True)
    assert submitted[-1] == submitted[0]
    assert len(worker.job_store.get_running()) == 1
    assert worker.job_store.get_finished()[0].wdir_names == ["m4"]
    assert worker.job_store.get_running()[0].attempt == 2
    worker.trainer.finished.add("m3")
    worker.inspect()
    assert len(worker.job_store.get_finished()) == 2
    assert len(worker.retrieve()) == 5
    assert worker.retrieve() == []
    assert len(worker.retrieve(include_retrieved=True)) == 5

def test_packed_launch_reports_child_failure(tmp_path):
    worker, _ = make_worker(tmp_path, 4)
    worker._submit = False
    worker.run(Dataset(), size=4)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gdp = bin_dir / "gdp"
    gdp.write_text('#!/bin/bash\ntouch launched\nif [[ "$PWD" == */m2 ]]; then exit 7; fi\n')
    gdp.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + ":" + os.environ["PATH"])
    script = worker.scheduler.script
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(["bash", str(script)], env=env, cwd=script.parent)
    assert result.returncode == 1
    assert all((tmp_path / f"m{i}/launched").exists() for i in range(4))


def test_restart_reconstructs_script_and_preserves_inputs(tmp_path):
    worker, submitted = make_worker(tmp_path, 4)
    worker.run(Dataset(), size=4)
    original = worker.job_store.get_running()[0]
    manifest = worker.metadata.inputs.path.read_bytes()
    script = worker.scheduler.script
    script.unlink()
    (tmp_path / "m0/trainer.yaml").unlink()
    resumed = TrainerBasedWorker(Trainer(), worker.scheduler, directory=tmp_path)
    resumed.scheduler.is_finished = lambda: True
    resumed.inspect(resubmit=True)
    assert resumed.metadata.inputs.path.read_bytes() == manifest
    assert script.exists()
    assert (tmp_path / "m0/trainer.yaml").exists()
    assert resumed.job_store.get_running()[0].uid == original.uid
    assert resumed.job_store.get_running()[0].attempt == 2


@pytest.mark.parametrize("concurrent", [1, 4])
def test_prepare_then_submit_uses_same_layout(tmp_path, concurrent):
    worker, submitted = make_worker(tmp_path, concurrent)
    worker._submit = False
    worker.run(Dataset(), size=4)
    assert not submitted
    assert len(list((tmp_path / "_meta/jobscripts").glob("*.script"))) == 4 // concurrent
    worker._submit = True
    worker.scheduler.is_finished = lambda: True
    worker.inspect(resubmit=True)
    assert len(submitted) == 4 // concurrent
    assert all(r.scheduler_job_id for r in worker.job_store.get_queued())


def test_changed_training_set_is_rejected(tmp_path):
    worker, submitted = make_worker(tmp_path, 4)
    worker.run(Dataset(), size=4)
    original = worker.metadata.inputs.path.read_bytes()
    with pytest.raises(ValueError, match="Calculation set conflict"):
        worker.run(Dataset(), size=3)
    assert worker.metadata.inputs.path.read_bytes() == original
    assert len(submitted) == 1


def test_legacy_training_layout_is_not_modified(tmp_path):
    path = tmp_path / "_slurm_jobs.json"
    path.write_text('{"_default": {}}')
    worker, _ = make_worker(tmp_path, 4)
    with pytest.raises(RuntimeError, match="Legacy training worker layout"):
        worker.inspect()
    assert path.read_text() == '{"_default": {}}'
    assert not (tmp_path / "_meta").exists()


def test_direct_training_uses_saved_model_configuration(monkeypatch, tmp_path):
    from gdpx.execution.schedulers.direct import DirectScheduler
    calls = []
    def run(configuration, directory):
        calls.append((configuration, directory))
    monkeypatch.setattr("gdpx.cli.train.run_trainer", run)
    worker = TrainerBasedWorker(Trainer(), DirectScheduler(), directory=tmp_path)
    worker.run(Dataset(), size=2)
    assert [directory.name for _, directory in calls] == ["m0", "m1"]
    assert all(configuration == directory / "trainer.yaml" for configuration, directory in calls)
    assert all(job.attempt == 1 for job in worker.job_store.get_queued())


def test_changed_trainer_options_are_rejected(tmp_path):
    worker, submitted = make_worker(tmp_path, 4)
    worker.run(Dataset(), size=4)
    worker.trainer.component_config = ComponentConfig("deepmd", "default", {"train_options": "--skip-neighbor-stat"})
    with pytest.raises(ValueError, match="Calculation set conflict"):
        worker.run(Dataset(), size=4)
    assert len(submitted) == 1
