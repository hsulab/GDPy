"""Shared generation progress and restart boundaries for population searches."""
import copy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from gdpx.exploration.generation import GenerationInfo, GenerationState, EvaluationStatus
from gdpx.exploration.persist.database import GlobalOptimisationDatabase
from gdpx.exploration.factory import create_expedition
from gdpx.execution.factory import create_worker
from gdpx.exploration.basin_hopping.chain import run_hopping_rounds
from gdpx.sampling import parse_operators


def atom(index=0):
    a = Atoms("Cu2", positions=[[5, 5, 5], [7.4 + index * .1, 5, 5]], tags=[1, 2], cell=[20] * 3)
    a.calc = SinglePointCalculator(a, energy=0., forces=np.zeros((2, 3)))
    return a


def relaxed(db, candidate, generation=0, extinct=0):
    candidate.info["key_value_pairs"] = dict(generation=generation, raw_score=0., extinct=extinct)
    db.add_relaxed_step(candidate)


def test_generation_progress_uses_unique_committed_ids(tmp_path):
    db = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    db.configure_generations(2, 3)
    assert db.get_generation_info() == GenerationInfo(0, GenerationState.BEG_OF_GEN, [], [])
    a, b = atom(), atom(1)
    db.add_unrelaxed_candidate(a, generation=0)
    db.add_unrelaxed_candidate(b, generation=0)
    db.set_generation_plan(0, {"stage": "complete"})
    assert db.get_generation_info().state is GenerationState.MID_OF_GEN
    relaxed(db, a)
    relaxed(db, a)  # Retry after a successful commit.
    assert db.connection.count(relaxed=1) == 1
    assert db.get_generation_info().unrelaxed_confids == [b.info["confid"]]
    # Multiple stored relaxed steps for one candidate still count only once.
    db.connection.write(a, relaxed=1, confid=a.info["confid"], generation=0, extinct=0)
    assert db.get_generation_number() == 0
    relaxed(db, b)
    assert db.get_generation_info(0).state is GenerationState.END_OF_GEN
    assert db.get_generation_number() == 1
    assert db.get_generation_info().converged(0)
    assert not db.get_generation_info().converged(1)


def test_extinction_waits_for_complete_generation(tmp_path):
    db = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    db.configure_generations(2, 1, use_extinct=True)
    a, b = atom(), atom(1)
    for candidate in [a, b]:
        db.add_unrelaxed_candidate(candidate, generation=0)
    relaxed(db, a, extinct=1)
    assert db.get_generation_info().state is GenerationState.MID_OF_GEN
    relaxed(db, b, extinct=1)
    assert db.get_generation_info().state is GenerationState.EXTINCTED
    assert db.get_generation_info().converged(50)


def bh_config(tmp_path, initial=2, generations=0):
    source = tmp_path / "seed.xyz"
    write(source, atom())
    runtime = {"potential": {"provider": "emt", "parameters": {}},
               "executor": {"provider": "ase", "method": "spc", "parameters": {}},
               "options": {"worker": "single"}}
    config = {"method": "basin_hopping", "recipe": {
        "population": {"periodic": False, "retained_size": 1,
                       "initial": {"total_size": initial, "builder_allocations": [{"builder": "random", "size": initial}]},
                       "generation": {"total_size": 2},
                       "builders": {"random": {"method": "read_stru", "fname": str(source)}}},
        "operators": [{"method": "move", "particles": ["Cu"], "max_disp": .05, "skip_distance_check": True}],
        "num_mcmoves": 2, "convergence": {"generation": generations},
        "random_seed": 7, "use_archive": False}}
    return config, runtime


def make_engine(config, runtime, directory):
    engine = create_expedition(copy.deepcopy(config))
    if isinstance(engine, list):
        engine = engine[0]
    engine.directory = directory
    engine.register_worker(create_worker(runtime))
    return engine


def test_bh_pending_batch_does_not_regenerate_inputs(tmp_path, monkeypatch):
    import gdpx.exploration.basin_hopping.engine as module
    config, runtime = bh_config(tmp_path)
    engine = make_engine(config, runtime, tmp_path / "run")
    original_execute = module.evaluate_batch
    submitted = []
    monkeypatch.setattr(module, "evaluate_batch", lambda candidates, *args, **kwargs:
                        submitted.extend(a.info["confid"] for a in candidates) or None)
    engine.run()
    db = GlobalOptimisationDatabase(engine.database_path)
    assert db.get_generation_info().state is GenerationState.MID_OF_GEN
    assert db.get_generation_number() == 0
    resumed = make_engine(config, runtime, engine.directory)
    monkeypatch.setattr(resumed.builders["random"], "run", lambda **kwargs: pytest.fail("regenerated inputs"))
    def execute(candidates, *args, **kwargs):
        assert [a.info["confid"] for a in candidates] == submitted
        return original_execute(candidates, *args, **kwargs)
    monkeypatch.setattr(module, "evaluate_batch", execute)
    resumed.run()
    assert db.connection.count(relaxed=0) == 2
    assert db.connection.count(relaxed=1) == 2
    assert resumed.read_convergence()


@pytest.mark.parametrize("method", ["bh", "ga"])
def test_partial_ingestion_restarts_without_duplicates(tmp_path, monkeypatch, method):
    if method == "bh":
        config, runtime = bh_config(tmp_path)
    else:
        import yaml
        path = Path(__file__).resolve().parents[2] / "examples/global_optimisation/cu13_emt.yaml"
        config = yaml.safe_load(path.read_text())
        runtime = config.pop("runtime")
        config["recipe"].update(convergence={"generation": 0}, use_archive=False)
    engine = make_engine(config, runtime, tmp_path / "run")
    original = GlobalOptimisationDatabase.add_relaxed_step
    calls = []
    def interrupt(db, candidate):
        original(db, candidate)
        calls.append(candidate.info["confid"])
        if len(calls) == 1:
            raise RuntimeError("interrupted after first result commit")
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_relaxed_step", interrupt)
    with pytest.raises(RuntimeError, match="after first result"):
        engine.run()
    path = engine.database_path if method == "bh" else engine.db_path
    db = GlobalOptimisationDatabase(path)
    assert db.get_generation_number() == 0
    assert db.get_generation_info().state is GenerationState.MID_OF_GEN
    resumed = make_engine(config, runtime, engine.directory)
    monkeypatch.setattr(resumed.builders["random"], "run", lambda **kwargs: pytest.fail("regenerated inputs"))
    resumed.run()
    assert db.connection.count(relaxed=1) == engine.population_config.init_size
    assert len(calls) == len(set(calls))
    assert db.get_generation_number() == 1
    assert resumed.read_convergence()
    if method == "ga":
        assert db.connection.count(substrate=True) == 1


@pytest.mark.parametrize("method", ["bh", "ga"])
def test_restart_between_generations_preserves_search_trajectory(tmp_path, monkeypatch, method):
    if method == "bh":
        config, runtime = bh_config(tmp_path, generations=2)
    else:
        import yaml
        path = Path(__file__).resolve().parents[2] / "examples/global_optimisation/cu13_emt.yaml"
        config = yaml.safe_load(path.read_text())
        runtime = config.pop("runtime")
        config["recipe"].update(convergence={"generation": 2}, use_archive=False)
    baseline = make_engine(config, runtime, tmp_path / "baseline")
    baseline.run()
    interrupted = make_engine(config, runtime, tmp_path / "interrupted")
    original = interrupted._irun
    def stop_before_second(*args, **kwargs):
        info = args[-1]
        if info.num == 2:
            raise RuntimeError("between generations")
        return original(*args, **kwargs)
    monkeypatch.setattr(interrupted, "_irun", stop_before_second)
    with pytest.raises(RuntimeError, match="between generations"):
        interrupted.run()
    resumed = make_engine(config, runtime, interrupted.directory)
    resumed.run()
    expected = read(baseline.directory / "results/all_candidates.xyz", ":")
    actual = read(resumed.directory / "results/all_candidates.xyz", ":")
    assert len(expected) == len(actual)
    for first, second in zip(expected, actual):
        np.testing.assert_allclose(first.positions, second.positions, atol=1e-10)
        assert first.get_potential_energy() == pytest.approx(second.get_potential_energy())
    assert baseline.random_streams.snapshot() == resumed.random_streams.snapshot()


def test_initial_batch_and_rng_checkpoint_commit_atomically(tmp_path, monkeypatch):
    config, runtime = bh_config(tmp_path)
    engine = make_engine(config, runtime, tmp_path / "run")
    original = GlobalOptimisationDatabase.add_unrelaxed_candidate
    calls = []
    def interrupt(db, candidate, *args, **kwargs):
        original(db, candidate, *args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("during batch transaction")
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_unrelaxed_candidate", interrupt)
    with pytest.raises(RuntimeError, match="batch transaction"):
        engine.run()
    db = GlobalOptimisationDatabase(engine.database_path)
    assert db.connection.count(relaxed=0) == 0
    resumed = make_engine(config, runtime, engine.directory)
    resumed.run()
    assert db.connection.count(relaxed=0) == 2
    assert db.connection.count(relaxed=1) == 2


def test_bh_endpoint_ingestion_resumes_without_evaluation(tmp_path, monkeypatch):
    config, runtime = bh_config(tmp_path, generations=1)
    engine = make_engine(config, runtime, tmp_path / "search")
    original = GlobalOptimisationDatabase.add_relaxed_step
    def interrupt(db, candidate):
        original(db, candidate)
        if candidate.info["key_value_pairs"]["generation"] == 1:
            raise RuntimeError("interrupted endpoint ingestion")
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_relaxed_step", interrupt)
    with pytest.raises(RuntimeError, match="endpoint ingestion"):
        engine.run()
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_relaxed_step", original)
    resumed = make_engine(config, runtime, engine.directory)
    monkeypatch.setattr(resumed.worker, "run", lambda *a, **k: pytest.fail("reevaluated endpoint"))
    resumed.run()
    db = GlobalOptimisationDatabase(engine.database_path)
    assert db.connection.count(relaxed=1, generation=1) == 2
    assert db.connection.count(relaxed=0, generation=1) == 2
    assert resumed.read_convergence()
    for row in db.connection.select(relaxed=1, generation=1):
        assert "chain" in row.data and len(row.data.parents) == 1


def test_bh_rejects_removed_mcworker_and_serial_checkpoints(tmp_path):
    config, runtime = bh_config(tmp_path, generations=1)
    legacy = copy.deepcopy(config)
    legacy["recipe"]["mcworker"] = runtime
    with pytest.raises(ValueError, match="mcworker.*top-level runtime"):
        create_expedition(legacy)
    engine = make_engine(config, runtime, tmp_path / "search")
    db = GlobalOptimisationDatabase(engine.database_path)
    db.set_generation_plan(1, dict(stage="hopping", parents=[1, 1]))
    with pytest.raises(ValueError, match="serial BH checkpoint"):
        engine._prepare_generation(db, 1, engine.directory / "tmp_folder/gen1")
