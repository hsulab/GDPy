import os
import subprocess
import yaml
from tinydb import TinyDB
from gdpx.execution.schedulers.slurm import SlurmScheduler
from gdpx.execution.workers.train import TrainerBasedWorker
from gdpx.providers import ComponentConfig

class Dataset:
    def as_dict(self):
        return {"name": "xyz", "dataset_path": "dataset"}

class Trainer:
    random_seed = 42
    component_config = ComponentConfig("deepmd", "default", {})
    def set_rng(self, seed):
        pass
    def read_convergence(self):
        return self.directory.name in self.finished
    finished = set()

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
    with TinyDB(tmp_path / "_slurm_jobs.json") as db:
        assert [r["wdir_names"] for r in db.all()] == [["m0", "m1", "m2", "m3"], ["m4"]]
    worker.scheduler.is_finished = lambda: True
    worker.trainer.finished = {"m0", "m1", "m2", "m4"}
    worker.inspect(resubmit=True)
    assert submitted[-1] == tmp_path / "m0" / "train.script"
    with TinyDB(tmp_path / "_slurm_jobs.json") as db:
        assert not db.all()[0].get("finished", False)
        assert db.all()[1]["finished"]
    worker.trainer.finished.add("m3")
    worker.inspect()
    with TinyDB(tmp_path / "_slurm_jobs.json") as db:
        assert all(r["finished"] for r in db.all())

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
    script = tmp_path / "m0/train.script"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(["bash", str(script)], env=env)
    assert result.returncode == 1
    assert all((tmp_path / f"m{i}/launched").exists() for i in range(4))
