"""Submission and inspection must not load calculation providers."""

import copy
import json

import pytest
from ase import Atoms
from ase.io import write

from gdpx.cli.explore import run_exploration
from gdpx.execution.schedulers.slurm import SlurmScheduler


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("remote_direct", [False, True])
def test_slurm_defers_model_loading_through_submission_and_resume(tmp_path, monkeypatch, finished, remote_direct):
    import gdpx.execution

    monkeypatch.chdir(tmp_path)
    write("seeds.xyz", Atoms("Cu", cell=[8, 8, 8], pbc=True))
    (tmp_path / "model.pt").write_bytes(b"must not be loaded on the submit host")
    params = {
        "method": "global_optimisation", "random_seed": 7,
        "system": {
            "periodic": True, "preserve_fragments": False,
            "initial": {"total_size": 1, "builder_allocations": [{"builder": "init", "size": 1}]},
            "generation": {"total_size": 1},
            "builders": {"init": {"method": "read_stru", "fname": "seeds.xyz"}},
        },
        "strategy": {
            "method": "simulated_annealing", "temperatures": [[600, 100]],
            "convergence": {"generation": 1},
            "minimisation": {"executor": {"provider": "ase", "method": "min",
                                            "parameters": {"stop": {"steps": 1, "fmax": .05}}}},
        },
        "runtime": {
            "potential": {"provider": "mace", "parameters": {"model": "./model.pt"}},
            "executor": {"provider": "ase", "method": "md", "parameters": {
                "setup": {"ensemble": {"method": "nvt", "temperature": 600},
                          "regulator": {"name": "nose_hoover_chain"}},
                "stop": {"steps": 1}}},
        },
        "scheduler": {"provider": "slurm"},
    }
    original = copy.deepcopy(params)
    loads = []

    def load(config):
        loads.append(config)
        raise RuntimeError("model loading reached")

    monkeypatch.setattr(gdpx.execution, "resolve_runtime", load)
    if remote_direct:
        # Direct execution over SSH also runs in another process. Exercise its
        # dispatch flags without opening a network connection in this test.
        monkeypatch.setattr(SlurmScheduler, "is_direct", True)
        monkeypatch.setattr(SlurmScheduler, "transport_name", "remote_test")
    submissions = []
    monkeypatch.setattr(SlurmScheduler, "submit", lambda self, **kw: submissions.append(self.job_name) or "123")
    monkeypatch.setattr(SlurmScheduler, "is_finished", lambda self: finished)
    directory = tmp_path / "_run"
    run_exploration(params, directory=directory)
    run_exploration(params, directory=directory)
    assert not loads
    assert len(submissions) == (3 if finished else 1)
    assert params == original
    saved_path, = (directory / "_meta").glob("exp-*.json")
    saved = json.loads(saved_path.read_text())
    assert saved["runtime"]["potential"]["parameters"]["model"] == str(tmp_path / "model.pt")
    assert saved["strategy"]["minimisation"]

    # The actual spawned process constructs its worker, even after chdir.
    monkeypatch.chdir(directory)
    with pytest.raises(RuntimeError, match="model loading reached"):
        run_exploration(saved, directory=directory, spawn="0")
    assert len(loads) == 1
    assert loads[0].potential.parameters["model"] == str(tmp_path / "model.pt")


def test_deferred_worker_materializes_once_at_execution(tmp_path, monkeypatch):
    from gdpx.execution import factory
    from gdpx.execution.workers.explore import run_exploration_in_commandline
    from gdpx.exploration.exploration import BaseExploration

    class Exploration(BaseExploration):
        def register_worker(self, worker):
            self.worker = worker

        def run(self):
            assert self.worker is sentinel

        def read_convergence(self):
            return True

        def get_workers(self):
            return [self.worker]

    sentinel = object()
    calls = []
    monkeypatch.setattr(factory, "create_worker", lambda config: calls.append(config) or sentinel)
    exploration = Exploration()
    exploration.register_runtime({"potential": {"provider": "emt"},
                                  "executor": {"provider": "ase", "method": "spc"}})
    assert not calls
    assert exploration.runtime_as_dict()["potential"]["provider"] == "emt"
    for _ in range(2):
        run_exploration_in_commandline(tmp_path, exploration)
    assert len(calls) == 1
