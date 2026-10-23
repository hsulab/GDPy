import copy
import json
from pathlib import Path

import pytest
import yaml
from ase.calculators.calculator import CalculationFailed

from gdpx.execution.factory import create_worker
from gdpx.exploration.factory import create_exploration
from gdpx.exploration.persist.database import GlobalOptimisationDatabase
from gdpx.providers.ase.driver import AseDriver


def setup_search(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[2] / "examples/global_optimisation"
    monkeypatch.chdir(examples / "explorations/simulated_annealing")
    config = yaml.safe_load(Path("cu8.yaml").read_text())
    runtime = yaml.safe_load((examples / "runtimes/emt_sa.yaml").read_text())
    runtime["executor"]["parameters"]["stop"]["steps"] = 1
    runtime["executor"]["parameters"]["output"]["trajectory"]["period"] = 1
    config["strategy"]["minimisation"]["runtime"]["executor"]["parameters"]["stop"] = {"steps": 100, "fmax": .1}
    config["strategy"]["temperatures"] = [[600, 100], [600, 100]]
    config["strategy"]["convergence"]["generation"] = 2
    config["system"]["generation"]["total_size"] = 2
    config["strategy"]["use_archive"] = False

    def construct():
        obj = create_exploration(copy.deepcopy(config))
        obj.directory = tmp_path
        obj.register_worker(create_worker(copy.deepcopy(runtime)))
        return obj
    return construct


@pytest.mark.parametrize("stage", ["min", "sim"])
@pytest.mark.parametrize("all_fail", [False, True])
def test_skip_failures_continue_and_resume(tmp_path, monkeypatch, stage, all_fail):
    construct = setup_search(tmp_path, monkeypatch)
    original = AseDriver._irun
    failed_attempts = []

    def fail(driver, *args, **kwargs):
        path = driver.directory
        if path.parent.name == stage and path.parent.parent.name == "gen1" and (
                all_fail or path.name == "cand0"):
            failed_attempts.append(str(path))
            raise CalculationFailed("injected candidate failure")
        return original(driver, *args, **kwargs)
    monkeypatch.setattr(AseDriver, "_irun", fail)
    # Interrupt after gen1 ingestion, before it is marked complete. Resume must
    # neither rerun failures nor duplicate already ingested successful minima.
    record = GlobalOptimisationDatabase.add_evaluated_candidate
    interrupted = []
    def interrupt(database, atoms, key):
        result = record(database, atoms, key)
        if key.startswith("sa:1:") and not interrupted:
            interrupted.append(True)
            raise RuntimeError("interrupted ingestion")
        return result
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_evaluated_candidate", interrupt)
    obj = construct()
    if all_fail:
        obj.run()
    else:
        with pytest.raises(RuntimeError, match="interrupted ingestion"):
            obj.run()
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_evaluated_candidate", record)
    resumed = construct()
    resumed.run()
    assert resumed.read_convergence()
    assert len(failed_attempts) == (2 if all_fail else 1)
    state = json.loads((tmp_path / "population.json").read_text())
    label = "minimisation" if stage == "min" else "dynamics"
    assert set(state["plans"]["1"]["failures"][label]) == ({"0", "1"} if all_fail else {"0"})
    db = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    rows = list(db.connection.select(relaxed=1))
    assert len(rows) == (4 if all_fail else 5)
    gen1 = [row for row in rows if row.generation == 1]
    if not all_fail:
        assert len(gen1) == 1
        assert gen1[0].data.replica == 1
        assert gen1[0].data.parents == [state["plans"]["1"]["parents"][1]]
    assert len([row for row in rows if row.generation == 2]) == 2
    resumed.run()
    assert len(failed_attempts) == (2 if all_fail else 1)


def test_all_initial_candidates_fail_stops_without_retry(tmp_path, monkeypatch):
    construct = setup_search(tmp_path, monkeypatch)
    attempts = []
    def fail(driver, *args, **kwargs):
        attempts.append(driver.directory.name)
        raise CalculationFailed("bad initial candidate")
    monkeypatch.setattr(AseDriver, "_irun", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="no surviving candidates"):
            construct().run()
    assert len(attempts) == 2


def test_ordinary_compute_still_raises_on_candidate_failure(tmp_path, monkeypatch):
    from ase import Atoms
    from gdpx.execution.workers.drive import DriverBatchError
    worker = create_worker({"potential": {"provider": "emt"},
        "executor": {"provider": "ase", "method": "min", "parameters": {"steps": 1}}},
        directory=tmp_path)
    def fail(*args, **kwargs):
        raise CalculationFailed("ordinary failure")
    monkeypatch.setattr(AseDriver, "_irun", fail)
    with pytest.raises(DriverBatchError):
        worker.run([Atoms("Cu", cell=[8]*3, pbc=True)])
