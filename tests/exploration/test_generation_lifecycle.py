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
from gdpx.exploration.sampling import parse_operators


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


@pytest.mark.parametrize("boundary", ["result", "round"])
def test_bh_trial_ingestion_is_idempotent(tmp_path, monkeypatch, boundary):
    import gdpx.exploration.basin_hopping.chain as chain
    config, runtime = bh_config(tmp_path, generations=1)
    baseline = make_engine(config, runtime, tmp_path / "baseline")
    baseline.run()
    engine = make_engine(config, runtime, tmp_path / "search")
    record = GlobalOptimisationDatabase.add_evaluated_candidate
    commit = chain._commit
    def interrupt_result(db, candidate, key):
        result = record(db, candidate, key)
        raise RuntimeError("interrupted result commit")
    def interrupt_round(directory, step, *args):
        if step == 1:
            raise RuntimeError("interrupted round commit")
        return commit(directory, step, *args)
    if boundary == "result":
        monkeypatch.setattr(GlobalOptimisationDatabase, "add_evaluated_candidate", interrupt_result)
    else:
        monkeypatch.setattr(chain, "_commit", interrupt_round)
    with pytest.raises(RuntimeError, match="interrupted"):
        engine.run()
    db = GlobalOptimisationDatabase(engine.database_path)
    assert db.get_generation_info().num == 1
    assert db.get_generation_info().state is GenerationState.MID_OF_GEN
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_evaluated_candidate", record)
    monkeypatch.setattr(chain, "_commit", commit)
    resumed = make_engine(config, runtime, engine.directory)
    resumed.run()
    assert db.connection.count(relaxed=1, generation=1) == 4
    assert db.connection.count(relaxed=0, generation=1) == 0
    assert resumed.read_convergence()
    expected = GlobalOptimisationDatabase(baseline.database_path)
    actual_rows = list(db.connection.select(relaxed=1, generation=1))
    for actual, wanted in zip(actual_rows, expected.connection.select(relaxed=1, generation=1)):
        assert actual.confid == wanted.confid
        assert dict(actual.data) == dict(wanted.data)
        np.testing.assert_allclose(actual.positions, wanted.positions)
        assert actual.energy == pytest.approx(wanted.energy)
    assert resumed.random_streams.snapshot() == baseline.random_streams.snapshot()
    assert len({row.evaluation_key for row in actual_rows}) == 4


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


def test_explicit_generation_results_complete_only_when_finalized(tmp_path):
    db = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    db.configure_generations(1, 1)
    initial = atom()
    db.add_unrelaxed_candidate(initial, generation=0)
    relaxed(db, initial)
    db.set_generation_plan(1, dict(stage="hopping", expected_confids=[]))
    ids = []
    for index in range(3):
        candidate = atom(index)
        candidate.info.update(key_value_pairs=dict(generation=1, raw_score=0.), data={})
        ids.append(db.add_evaluated_candidate(candidate, f"trial-{index}"))
        assert db.get_generation_info().state is GenerationState.MID_OF_GEN
        assert db.get_generation_number() == 1
    db.set_generation_plan(1, dict(stage="complete", expected_confids=ids + [999]))
    assert db.get_generation_info().state is GenerationState.MID_OF_GEN
    db.set_generation_plan(1, dict(stage="complete", expected_confids=ids))
    assert db.get_generation_info(1).state is GenerationState.END_OF_GEN
    assert db.get_generation_number() == 2
    db.set_generation_plan(2, dict(stage="complete", expected_confids=[]))
    assert db.get_generation_number() == 3


class HistoryWorker:
    """Deterministic evaluated endpoints with controllable uphill acceptance."""
    def __init__(self, energies):
        self.energies = iter(energies)
        self.calls = 0
    def run(self, frames):
        energy = next(self.energies)
        self.calls += 1
        self.results = []
        for frame in frames:
            result = frame.copy()
            result.info = copy.deepcopy(frame.info)
            result.calc = SinglePointCalculator(result, energy=energy, forces=np.zeros((len(frame), 3)))
            self.results.append([result])
    def inspect(self, **kwargs):
        pass
    def get_number_of_running_jobs(self):
        return 0
    def retrieve(self, **kwargs):
        return self.results


@pytest.mark.parametrize("accept_uphill", [False, True])
def test_every_minimum_is_available_to_next_population(tmp_path, accept_uphill):
    config, runtime = bh_config(tmp_path, generations=2 if accept_uphill else 1)
    config["recipe"]["population"]["comparator"] = {"method": "atoms"}
    config["recipe"]["population"]["retained_size"] = 1 if accept_uphill else 10
    config["recipe"]["operators"][0]["temperature"] = 1e12 if accept_uphill else 1e-6
    engine = make_engine(config, runtime, tmp_path / "search")
    worker = HistoryWorker([0., -10., -1., -2., -3.])
    engine.register_worker(worker)
    engine.run()
    db = GlobalOptimisationDatabase(engine.database_path)
    rows = list(db.connection.select(relaxed=1, generation=1))
    assert len(rows) == 4
    earlier = [row for row in rows if row.data["round"] == 1]
    later = [row for row in rows if row.data["round"] == 2]
    assert all(row.data.accepted is accept_uphill for row in later)
    assert {row.data.parents[0] for row in later} == {row.confid for row in earlier}
    assert db.connection.count(relaxed=0, generation=1) == 0
    if accept_uphill:
        # Generation 2 starts at an intermediate -10 minimum, not a -1 endpoint.
        assert set(db.get_generation_plan(2)["parents"]) <= {row.confid for row in earlier}
        assert set(db.get_generation_plan(2)["parents"]) <= set(db.get_generation_plan(2)["population"])
        assert len(db.get_generation_plan(2)["population"]) == 1
        assert worker.calls == 5
    else:
        engine.population.refresh(db)
        assert {row.confid for row in later} <= {a.info["confid"] for a in engine.population.candidates}
        assert worker.calls == 3
    frames = read(engine.directory / "results/all_candidates.xyz", ":")
    assert len(frames) == db.connection.count(relaxed=1)


@pytest.mark.parametrize("invalid", [False, True])
def test_bh_zero_result_generation_completes_and_reports(tmp_path, monkeypatch, invalid):
    from gdpx.exploration.sampling.proposal import MoveProposal
    config, runtime = bh_config(tmp_path, generations=2)
    if not invalid:
        config["recipe"]["num_mcmoves"] = 0
    engine = make_engine(config, runtime, tmp_path / "search")
    if invalid:
        def reject(atoms, rng):
            proposal = MoveProposal(atoms)
            proposal.valid = False
            proposal.rollback()
            return proposal
        monkeypatch.setattr(engine.operators[0], "propose", reject)
    worker = HistoryWorker([0.])
    engine.register_worker(worker)
    engine.run()
    db = GlobalOptimisationDatabase(engine.database_path)
    assert engine.read_convergence()
    assert db.get_generation_number() == 3
    assert db.connection.count(relaxed=1) == 2
    assert worker.calls == 1
    assert db.get_generation_plan(1)["expected_confids"] == []
    assert (engine.directory / "results/pop.png").exists()


@pytest.mark.parametrize("convergence", [None, {}, {"generation": 1}])
def test_bh_default_generation_and_serialization(tmp_path, convergence):
    config, runtime = bh_config(tmp_path)
    if convergence is None:
        config["recipe"].pop("convergence")
    else:
        config["recipe"]["convergence"] = convergence
    engine = make_engine(config, runtime, tmp_path / "search")
    assert engine.convergence == {"generation": 1}
    serialized = engine.as_dict()
    assert serialized["recipe"]["convergence"] == {"generation": 1}
    serialized.pop("runtime")
    assert create_expedition(serialized).convergence == {"generation": 1}
    engine.run()
    db = GlobalOptimisationDatabase(engine.database_path)
    assert db.get_generation_number() == 2
    assert db.connection.count(relaxed=1, generation=1) == 4
    assert db.get_generation_plan(2) is None
    if convergence is not None:
        assert config["recipe"]["convergence"] == convergence


@pytest.mark.parametrize("generation", [-1, 1.5, True, "1", None])
def test_bh_rejects_invalid_generation_limits(tmp_path, generation):
    config, _ = bh_config(tmp_path)
    config["recipe"]["convergence"] = {"generation": generation}
    with pytest.raises(ValueError, match="convergence.generation.*non-negative integer"):
        create_expedition(config)


def test_bh_default_generation_resumes_without_reselecting_starts(tmp_path, monkeypatch):
    import gdpx.exploration.basin_hopping.chain as chain
    config, runtime = bh_config(tmp_path)
    config["recipe"].pop("convergence")
    baseline = make_engine(config, runtime, tmp_path / "baseline")
    baseline.run()
    interrupted = make_engine(config, runtime, tmp_path / "interrupted")
    commit = chain._commit
    def stop(directory, step, *args):
        commit(directory, step, *args)
        if step == 1:
            raise RuntimeError("stop after first round")
    monkeypatch.setattr(chain, "_commit", stop)
    with pytest.raises(RuntimeError, match="first round"):
        interrupted.run()
    monkeypatch.setattr(chain, "_commit", commit)
    resumed = make_engine(config, runtime, interrupted.directory)
    monkeypatch.setattr(resumed.start_selector, "select", lambda *a: pytest.fail("reselected starts"))
    resumed.run()
    assert resumed.read_convergence()
    expected = read(baseline.directory / "results/all_candidates.xyz", ":")
    actual = read(resumed.directory / "results/all_candidates.xyz", ":")
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.positions, b.positions)
    assert resumed.random_streams.snapshot() == baseline.random_streams.snapshot()


def extinction_engine(config, runtime, directory, worker, callback):
    engine = make_engine(config, runtime, directory)
    engine.register_worker(worker)
    engine.population_config.extinct_callbacks = [callback]
    engine.population_config.use_extinct = True
    engine.population.use_extinct = True
    return engine


@pytest.mark.parametrize("trial_energy,extinct", [(-1., False), (100., False), (-1., True), (100., True)])
def test_bh_extinction_requires_mc_acceptance(tmp_path, monkeypatch, trial_energy, extinct):
    config, runtime = bh_config(tmp_path, generations=1)
    engine = extinction_engine(config, runtime, tmp_path / "search", HistoryWorker([0., trial_energy, 0.]),
                               lambda a: int(extinct and a.get_potential_energy() == trial_energy))
    selections = []
    select = engine.start_selector.select
    def record_selection(population, count):
        selections.append([a.info["confid"] for a in population.candidates])
        return select(population, count)
    monkeypatch.setattr(engine.start_selector, "select", record_selection)
    engine.run()
    db = GlobalOptimisationDatabase(engine.database_path)
    first = [r for r in db.connection.select(generation=1) if r.data["round"] == 1]
    second = [r for r in db.connection.select(generation=1) if r.data["round"] == 2]
    terminated = trial_energy < 0 and extinct
    assert len(selections) == (2 if terminated else 1)
    assert len(first) == len(second) == 2
    assert all(r.data.accepted == (trial_energy < 0) and r.extinct == int(extinct) for r in first)
    assert all(r.data.outcome == ("extinct" if terminated else ("accepted" if trial_energy < 0 else "rejected"))
               for r in first)
    assert all(r.data.segment == int(terminated) for r in second)
    if terminated:
        assert all(db.connection.get(confid=r.data.start_parent, relaxed=1).generation == 0 for r in second)
        assert all(db.connection.get(confid=i, relaxed=1).extinct == 0 for i in selections[-1])
    assert db.connection.count(generation=1, relaxed=1) == 4


class RoundEnergyWorker(HistoryWorker):
    def __init__(self):
        self.calls = 0
    def run(self, frames):
        if self.directory.name == "gen0":
            energies = [0., 0.]
        elif self.directory.name == "round-000001":
            energies = [-10., -2.]
        else:
            energies = [-1., -20.]
        self.calls += 1
        self.results = []
        for frame, energy in zip(frames, energies):
            result = frame.copy()
            result.info = copy.deepcopy(frame.info)
            result.calc = SinglePointCalculator(result, energy=energy, forces=np.zeros((len(frame), 3)))
            self.results.append([result])


def test_bh_restart_uses_complete_round_and_marks_trajectory(tmp_path, monkeypatch):
    config, runtime = bh_config(tmp_path, generations=1)
    config["recipe"]["operators"][0]["temperature"] = 1e12
    engine = extinction_engine(config, runtime, tmp_path / "search", RoundEnergyWorker(),
                               lambda a: int(a.get_potential_energy() < -5))
    select = engine.start_selector.select
    pools = []
    def record(population, count):
        pools.append([(a.info["confid"], a.get_potential_energy()) for a in population.candidates])
        return select(population, count)
    monkeypatch.setattr(engine.start_selector, "select", record)
    engine.run()
    assert len(pools) == 2  # Initial selection and one replacement; no restart after final round.
    assert pools[1][0][1] == -2.
    db = GlobalOptimisationDatabase(engine.database_path)
    records = {r.evaluation_key: r for r in db.connection.select(generation=1)}
    source = records["bh:1:1:1"].confid
    replacement_trial = records["bh:1:0:2"]
    assert replacement_trial.data.start_parent == source
    assert replacement_trial.data.parents == [source]
    assert replacement_trial.data.segment == 1
    assert records["bh:1:1:2"].data.outcome == "extinct"
    from gdpx.exploration.basin_hopping import export_trajectories
    paths = export_trajectories(engine.directory / "tmp_folder/gen1/rounds", engine.directory / "export")
    trajectory = read(paths[0], ":")
    assert [a.info["event"] for a in trajectory] == ["start", "restart", "accepted"]
    assert trajectory[1].info["source_confid"] == source
    assert trajectory[1].info["segment"] == 1
    assert trajectory[1].get_potential_energy() == -2.
    # A replacement is separately loaded, not aliased to the other active chain.
    assert replacement_trial.data.parents == records["bh:1:1:2"].data.parents


@pytest.mark.parametrize("boundary", ["result", "selection", "round", "pending"])
def test_bh_extinction_restart_preserves_all_random_streams(tmp_path, monkeypatch, boundary):
    import gdpx.exploration.basin_hopping.chain as chain
    config, runtime = bh_config(tmp_path, generations=1)
    config["recipe"]["operators"][0]["temperature"] = 1e12
    def make(directory):
        return extinction_engine(config, runtime, directory, RoundEnergyWorker(),
                                 lambda a: int(a.get_potential_energy() < -5))
    baseline = make(tmp_path / "baseline")
    baseline.run()
    interrupted = make(tmp_path / "interrupted")
    with monkeypatch.context() as patch:
        if boundary == "result":
            record = GlobalOptimisationDatabase.add_evaluated_candidate
            def fail(db, atoms, key):
                record(db, atoms, key)
                raise RuntimeError("interrupted result")
            patch.setattr(GlobalOptimisationDatabase, "add_evaluated_candidate", fail)
        elif boundary == "selection":
            select = interrupted.start_selector.select
            calls = []
            def fail(population, count):
                result = select(population, count)
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("interrupted selection")
                return result
            patch.setattr(interrupted.start_selector, "select", fail)
        elif boundary == "round":
            commit = chain._commit
            def fail(directory, step, *args):
                commit(directory, step, *args)
                if step == 1:
                    raise RuntimeError("interrupted round")
            patch.setattr(chain, "_commit", fail)
        else:
            patch.setattr(interrupted.worker, "get_number_of_running_jobs",
                          lambda: int(interrupted.worker.directory.name == "round-000002"))
        if boundary == "pending":
            interrupted.run()
            assert not interrupted.read_convergence()
        else:
            with pytest.raises(RuntimeError, match="interrupted"):
                interrupted.run()
    resumed = make(interrupted.directory)
    resumed.run()
    assert resumed.read_convergence()
    assert baseline.random_streams.snapshot() == resumed.random_streams.snapshot()
    expected = GlobalOptimisationDatabase(baseline.database_path)
    actual = GlobalOptimisationDatabase(resumed.database_path)
    assert actual.connection.count(generation=1) == 4
    for a, b in zip(actual.connection.select(generation=1), expected.connection.select(generation=1)):
        assert a.confid == b.confid and dict(a.data) == dict(b.data)
        np.testing.assert_array_equal(a.positions, b.positions)
    from gdpx.exploration.basin_hopping import export_trajectories
    for engine in (baseline, resumed):
        assert not (engine.directory / "tmp_folder/gen1/mctrajs").exists()
        export_trajectories(engine.directory / "tmp_folder/gen1/rounds", engine.directory / "export")
    for index in range(2):
        name = f"export/mc-{index:04d}.xyz"
        first = read(baseline.directory / name, ":")
        second = read(resumed.directory / name, ":")
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert {k: v for k, v in a.info.items() if k != "unique_id"} == {k: v for k, v in b.info.items() if k != "unique_id"}
            np.testing.assert_array_equal(a.positions, b.positions)


def test_bh_empty_restart_pool_terminates_search(tmp_path, monkeypatch):
    config, runtime = bh_config(tmp_path, generations=3)
    engine = extinction_engine(config, runtime, tmp_path / "search", HistoryWorker([0., -1.]),
                               lambda a: int(a.get_potential_energy() < 0))
    select = engine.start_selector.select
    calls = []
    def empty_after_start(population, count):
        calls.append(1)
        return select(population, count) if len(calls) == 1 else []
    monkeypatch.setattr(engine.start_selector, "select", empty_after_start)
    engine.run()
    db = GlobalOptimisationDatabase(engine.database_path)
    assert db.get_generation_info().state is GenerationState.EXTINCTED
    assert engine.read_convergence()
    assert db.get_generation_plan(1)["termination_reason"] == "extinct"
    assert db.connection.count(generation=1) == 2
    assert db.get_generation_plan(2) is None


@pytest.mark.parametrize('replace', [False, True])
def test_bh_selection_config_roundtrip(tmp_path, replace):
    config, runtime = bh_config(tmp_path)
    config['recipe']['selection'] = {'replace': replace}
    engine = make_engine(config, runtime, tmp_path / 'run')
    assert engine.start_selector.replace is replace
    exported = engine.as_dict()
    assert exported['recipe']['selection'] == {'replace': replace}
    restored_runtime = exported.pop('runtime')
    restored = make_engine(exported, restored_runtime, tmp_path / 'restored')
    assert restored.start_selector.replace is replace


@pytest.mark.parametrize('selection', [True, [], {'replace': 'false'}, {'replace': 0}, {'replacement': False}])
def test_bh_selection_invalid_config(tmp_path, selection):
    config, runtime = bh_config(tmp_path)
    config['recipe']['selection'] = selection
    with pytest.raises((TypeError, ValueError), match='selection'):
        make_engine(config, runtime, tmp_path / 'run')


@pytest.mark.parametrize('replace', [False, True])
def test_bh_generation_and_extinction_share_selection_policy(tmp_path, monkeypatch, replace):
    config, runtime = bh_config(tmp_path, generations=1)
    config['recipe']['population']['retained_size'] = 2
    config['recipe']['selection'] = {'replace': replace}
    engine = extinction_engine(config, runtime, tmp_path / 'run', HistoryWorker([0., -1., 0.]),
                               lambda a: int(a.get_potential_energy() == -1.))
    monkeypatch.setattr(engine.population.comparator, 'looks_like', lambda a, b: False)
    selections = []
    original = engine.start_selector.select
    def select(population, count):
        result = original(population, count)
        selections.append(([a.info['confid'] for a in result], len(population.candidates)))
        return result
    monkeypatch.setattr(engine.start_selector, 'select', select)
    engine.run()
    assert len(selections) == 2  # Generation starts, then both chains restart.
    assert all(available == 2 and len(ids) == 2 for ids, available in selections)
    if not replace:
        assert all(len(set(ids)) == 2 for ids, _ in selections)
    assert engine.start_selector.replace is replace
