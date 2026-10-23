"""Snapshot selection, provenance, and restart for population annealing."""
import copy
import json

import pytest
from ase import Atoms
from ase.io import read, write

from gdpx.analysis.selectors.selector import BaseSelector
from gdpx.execution.factory import create_worker
from gdpx.exploration.factory import create_exploration
from gdpx.exploration.persist.database import GlobalOptimisationDatabase
from gdpx.exploration.simulated_annealing.population import PopulationAnnealing


@pytest.fixture
def search(tmp_path):
    write(tmp_path / "seed.xyz", Atoms("Cu2", positions=[[4, 5, 5], [6.4, 5, 5]],
                                      cell=[10]*3, pbc=True))
    recipe = dict(method="global_optimisation", random_seed=7, system=dict(
        periodic=True, preserve_fragments=False, retained_size=1,
        builders={"init": {"method": "read_stru", "fname": str(tmp_path / "seed.xyz")}},
        initial={"total_size": 1, "builder_allocations": [{"builder": "init", "size": 1}]},
        generation={"total_size": 2}), strategy=dict(
        method="simulated_annealing", temperatures=[[600, 300]], minimisation={
            "runtime": {"executor": {"provider": "ase", "method": "min", "parameters": {
                "stop": {"steps": 100, "fmax": .1}}}}}))
    runtime = dict(potential={"provider": "emt"}, executor=dict(
        provider="ase", method="md", parameters=dict(
            ensemble={"method": "nvt", "temperature": 600},
            setup={"timestep": .1, "velocities": {"remove_rotation": False},
                   "regulator": {"name": "berendsen", "parameters": {"Tdamp": 10}}},
            stop={"steps": 4}, output={"trajectory": {"period": 1}})))

    def construct(selector=None):
        config = copy.deepcopy(recipe)
        if selector is not None:
            config["strategy"]["minimisation"]["selector"] = selector
        obj = create_exploration(config)
        obj.directory = tmp_path / "run"
        obj.register_worker(create_worker(copy.deepcopy(runtime)))
        return obj
    return construct


@pytest.mark.parametrize("selector,expected", [
    (None, [4]),
    ({"method": "locate", "indices": ":"}, [0, 1, 2, 3, 4]),
    ({"method": "locate", "indices": "1 3 -1"}, [1, 3, 4]),
    ({"method": "interval", "period": 2, "include_last": True}, [0, 2, 4]),
    ({"method": "locate", "indices": "99"}, []),
])
def test_selects_per_trajectory_and_records_minima(search, selector, expected):
    obj = search(selector)
    obj.run()
    root = obj.directory
    assert not (root / "gen0/sel").exists()
    records = json.loads((root / "gen1/sel/selection.json").read_text())["snapshots"]
    assert [(r["replica"], r["frame_index"]) for r in records] == [
        (replica, index) for replica in range(2) for index in expected]
    assert all(r["md_step"] == r["frame_index"] for r in records)
    if records:
        frames = read(root / "gen1/sel/selected.xyz", ":")
        assert len(frames) == len(records)
        assert all("momenta" not in a.arrays for a in frames)
    else:
        assert (root / "gen1/sel/selected.xyz").read_text() == ""
    rows = list(GlobalOptimisationDatabase(root / "candidates.db").connection.select(relaxed=1))
    assert len(rows) == 1 + 2 * len(expected)
    minima = [r for r in rows if r.generation == 1]
    assert [(r.data.replica, r.data.frame_index) for r in minima] == [
        (r["replica"], r["frame_index"]) for r in records]
    assert all(r.data.parents == [records[i]["parent"]] for i, r in enumerate(minima))
    assert len(obj.population.candidates) == 1


@pytest.mark.parametrize("selection", [
    {"method": "random", "number": 2},
    [{"method": "locate", "indices": ":"}, {"method": "random", "number": 2}],
])
def test_restart_reuses_selection_after_partial_ingestion(search, monkeypatch, selection):
    obj = search(selection)
    original = GlobalOptimisationDatabase.add_evaluated_candidate
    def interrupt(database, atoms, key):
        result = original(database, atoms, key)
        if key.startswith("sa:1:"):
            raise RuntimeError("interrupted ingestion")
        return result
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_evaluated_candidate", interrupt)
    with pytest.raises(RuntimeError, match="interrupted ingestion"):
        obj.run()
    saved = (obj.directory / "gen1/sel/selected.xyz").read_bytes()
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_evaluated_candidate", original)
    def unexpected(*args, **kwargs):
        raise AssertionError("Persisted selection must bypass MD and selection")
    monkeypatch.setattr(BaseSelector, "select", unexpected)
    monkeypatch.setattr(PopulationAnnealing, "_cooling", unexpected)
    resumed = search(selection)
    resumed.run()
    resumed.run()
    assert resumed.read_convergence()
    assert (obj.directory / "gen1/sel/selected.xyz").read_bytes() == saved
    rows = list(GlobalOptimisationDatabase(obj.directory / "candidates.db").connection.select(relaxed=1))
    assert len(rows) == 5


def test_failed_snapshot_does_not_discard_siblings(search, monkeypatch):
    from ase.calculators.calculator import CalculationFailed
    from gdpx.providers.ase.driver import AseDriver
    original = AseDriver._irun
    attempts = []
    def fail(driver, *args, **kwargs):
        if driver.directory.parts[-3:] == ("gen1", "min", "cand0"):
            attempts.append(1)
            raise CalculationFailed("one snapshot failed")
        return original(driver, *args, **kwargs)
    monkeypatch.setattr(AseDriver, "_irun", fail)
    obj = search({"method": "locate", "indices": "1 3"})
    obj.run()
    obj.run()
    assert len(attempts) == 1
    rows = list(GlobalOptimisationDatabase(obj.directory / "candidates.db").connection.select(relaxed=1))
    assert len(rows) == 4  # Initial minimum plus three successful snapshots.
    assert {(r.data.replica, r.data.frame_index) for r in rows if r.generation == 1} == {
        (0, 3), (1, 1), (1, 3)}


def test_rejects_old_minimisation_shape(search):
    obj = search()
    recipe = obj.as_dict()
    runtime = recipe.pop("runtime")
    recipe["strategy"]["minimisation"] = {"executor": recipe["strategy"]["minimisation"]["runtime"]["executor"]}
    with pytest.raises(ValueError, match="requires runtime"):
        create_exploration(recipe)


def test_selection_configuration_changes_require_new_directory(search):
    obj = search()
    obj.run()
    with pytest.raises(ValueError, match="configuration changed"):
        search({"method": "locate", "indices": ":"}).run()


def with_minimum_runtime(search, **overrides):
    obj = search()
    recipe = obj.as_dict()
    md_runtime = recipe.pop("runtime")
    recipe["strategy"]["minimisation"]["runtime"].update(overrides)
    rebuilt = create_exploration(recipe)
    rebuilt.directory = obj.directory
    rebuilt.register_worker(create_worker(md_runtime))
    return rebuilt


def test_minimum_runtime_inherits_potential_and_overrides_dispatch(search):
    obj = search()
    recipe = obj.as_dict()
    md_runtime = recipe.pop("runtime")
    md_runtime["dispatch"].update(batch_size=7, retain_info=True)
    md_runtime["scheduler"] = {"provider": "slurm"}
    recipe["strategy"]["minimisation"]["runtime"]["dispatch"] = {"batch_size": 2}
    original = copy.deepcopy((recipe, md_runtime))
    rebuilt = create_exploration(recipe)
    rebuilt.register_worker(create_worker(md_runtime))
    minimum = rebuilt._minimum_runtime()
    assert minimum["potential"] == rebuilt.worker.runtime.config.potential.to_dict()
    assert minimum["dispatch"]["batch_size"] == 2
    assert minimum["dispatch"]["retain_info"] is True
    assert minimum.get("scheduler", {}).get("provider", "direct") == "direct"
    assert (recipe, md_runtime) == original
    overridden = with_minimum_runtime(search, scheduler={"provider": "slurm"})
    assert overridden._minimum_runtime()["scheduler"]["provider"] == "slurm"


def test_minimum_runtime_uses_own_potential_for_every_generation(search):
    from ase.calculators.lj import LennardJones
    potential = {"provider": "ase", "parameters": {"method": "lj", "sigma": 2.0, "epsilon": .5}}
    obj = with_minimum_runtime(search, potential=potential)
    assert obj.worker.runtime.config.potential.provider == "emt"
    assert obj._minimum_runtime()["potential"]["provider"] == "ase"
    obj.run()
    rows = list(GlobalOptimisationDatabase(obj.directory / "candidates.db").connection.select(relaxed=1))
    assert {row.generation for row in rows} == {0, 1}
    assert len(rows) == 3
    for row in rows:
        atoms = row.toatoms()
        atoms.calc = LennardJones(sigma=2.0, epsilon=.5)
        assert row.energy == pytest.approx(atoms.get_potential_energy(), abs=1e-6)
    saved = obj.as_dict()
    md_runtime = saved.pop("runtime")
    rebuilt = create_exploration(saved)
    rebuilt.register_worker(create_worker(md_runtime))
    assert rebuilt._minimum_runtime() == obj._minimum_runtime()


@pytest.mark.parametrize("overrides,message", [
    ({"dispatch": {"share_workdir": True}}, "separate candidate directories"),
    ({"dispatch": {"worker": "single"}}, "dispatch.worker: batch"),
    ({"dispatch": {"batch_size": 0}}, "positive integer"),
    ({"unexpected": True}, "Unknown runtime fields"),
])
def test_minimum_runtime_rejects_unsupported_configuration(search, overrides, message):
    with pytest.raises(ValueError, match=message):
        with_minimum_runtime(search, **overrides)
