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
                                      cell=[10]*3, pbc=True, tags=[1, 2]))
    recipe = dict(method="global_optimisation", random_seed=7, system=dict(
        periodic=True, preserve_fragments=False, retained_size=1,
        builders={"init": {"method": "read_stru", "fname": str(tmp_path / "seed.xyz")}},
        initial={"total_size": 1, "builder_allocations": [{"builder": "init", "size": 1}]},
        generation={"total_size": 2}), strategy=dict(
        method="simulated_annealing", temperatures=[[600, 300]], minimisation={
            "executor": {"provider": "ase", "method": "min", "parameters": {
                "stop": {"steps": 100, "fmax": .1}}}}))
    runtime = dict(potential={"provider": "emt"}, executor=dict(
        provider="ase", method="md", parameters=dict(
            setup={"ensemble": {"method": "nvt", "temperature": 600}, "timestep": .1, "velocities": {"remove_rotation": False},
                   "regulator": {"name": "berendsen", "parameters": {"Tdamp": 10}}},
            stop={"steps": 4}, output={"trajectory": {"period": 1}})))

    simulation = runtime["executor"]
    simulation["parameters"]["setup"]["ensemble"].pop("temperature")
    runtime["executor"] = recipe["strategy"].pop("minimisation")["executor"]
    recipe["strategy"]["simulation"] = simulation

    def construct(selector=None):
        config = copy.deepcopy(recipe)
        if selector is not None:
            config["strategy"]["selection"] = selector
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
    if selector is not None:
        selector = dict(selector, group_by="axis 0")
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
    assert len(rows) == 3


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
    obj = search({"method": "locate", "indices": "1 3", "group_by": "axis 0"})
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
    recipe["strategy"]["minimisation"] = {"runtime": runtime}
    with pytest.raises(ValueError, match="Move strategy.minimisation"):
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
    md_runtime.update(overrides)
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
    md_runtime["dispatch"]["batch_size"] = 2
    original = copy.deepcopy((recipe, md_runtime))
    rebuilt = create_exploration(recipe)
    rebuilt.register_worker(create_worker(md_runtime))
    minimum = rebuilt._minimum_runtime()
    assert minimum["potential"] == rebuilt.worker.runtime.config.potential.to_dict()
    assert minimum["dispatch"]["batch_size"] == 2
    assert minimum["dispatch"]["retain_info"] is True
    assert minimum.get("scheduler", {}).get("provider", "direct") == "slurm"
    assert (recipe, md_runtime) == original
    overridden = with_minimum_runtime(search, scheduler={"provider": "slurm"})
    assert overridden._minimum_runtime()["scheduler"]["provider"] == "slurm"


def test_minimum_runtime_uses_own_potential_for_every_generation(search):
    from ase.calculators.lj import LennardJones
    potential = {"provider": "ase", "parameters": {"method": "lj", "sigma": 2.0, "epsilon": .5}}
    obj = with_minimum_runtime(search, potential=potential)
    assert obj.worker.runtime.config.potential.provider == "ase"
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


@pytest.mark.parametrize("tags", [[1, 2], [0, 7]])
def test_candidate_tags_survive_backend_output_and_restart(search, monkeypatch, tags):
    import numpy as np
    from gdpx.execution.workers.drive import DriverBasedWorker

    obj = search()
    seed_path = obj.directory.parent / "seed.xyz"
    seed = read(seed_path)
    if tags is not None:
        seed.set_tags(tags)
        write(seed_path, seed)
    expected = [1, 2] if tags is None else tags
    retrieve = DriverBasedWorker.retrieve

    def without_tags(worker, *args, **kwargs):
        trajectories = retrieve(worker, *args, **kwargs)
        for trajectory in trajectories:
            for frame in trajectory:
                frame.arrays.pop("tags", None)
        return trajectories

    monkeypatch.setattr(DriverBasedWorker, "retrieve", without_tags)
    obj.run()
    search().run()
    paths = [obj.directory / "gen0/inputs.xyz",
             obj.directory / "gen1/inputs.xyz",
             obj.directory / "gen1/sel/selected.xyz",
             obj.directory / "results/all_candidates.xyz"]
    for path in paths:
        for frame in read(path, ":"):
            np.testing.assert_array_equal(frame.get_tags(), expected)
    for frame in obj.get_minimised_candidates():
        np.testing.assert_array_equal(frame.get_tags(), expected)


@pytest.mark.parametrize("selection,expected", [
    ({"method": "locate", "indices": "1 -1"}, [(2, 1), (7, 2)]),
    ({"method": "locate", "indices": "-1", "group_by": "axis 0"}, [(2, 1), (7, 2)]),
    ({"method": "locate", "indices": ":"},
     [(2, 0), (2, 1), (7, 0), (7, 1), (7, 2)]),
    ([{"method": "locate", "indices": "-1", "group_by": "axis 0"},
      {"method": "locate", "indices": "0"}], [(2, 1)]),
])
def test_generation_selection_preserves_ragged_trajectory_provenance(search, selection, expected):
    obj = search(selection)
    trajectories = {7: [Atoms("Cu") for _ in range(3)],
                    2: [Atoms("Cu") for _ in range(2)]}
    plan = {"seed": 13, "parents": list(range(10, 18))}
    frames, records = obj._select_snapshots(1, trajectories, plan)
    assert [(r["replica"], r["frame_index"]) for r in records] == expected
    assert [r["parent"] for r in records] == [plan["parents"][r] for r, _ in expected]
    assert len(frames) == len(expected)
    assert all(r["md_step"] == -1 for r in records)


@pytest.mark.parametrize("group_by,number", [(None, 2), ("axis 0", 4)])
def test_random_selection_scope(search, group_by, number):
    obj = search({"method": "random", "number": 2, "group_by": group_by})
    trajectories = {i: [Atoms("Cu") for _ in range(3)] for i in range(2)}
    frames, records = obj._select_snapshots(1, trajectories, {"seed": 13, "parents": [0, 1]})
    assert len(frames) == len(records) == number
    if group_by is not None:
        assert [r["replica"] for r in records] == [0, 0, 1, 1]


def test_selection_with_no_successful_trajectories(search):
    obj = search()
    assert obj._select_snapshots(1, {}, {"seed": 13, "parents": []}) == ([], [])
    assert obj._select_snapshots(1, {}, {}) == ([], [])


def test_rejects_nested_selector_with_migration_message(search):
    recipe = search().as_dict()
    recipe.pop("runtime")
    recipe["strategy"]["simulation"]["selector"] = {"method": "random"}
    with pytest.raises(ValueError, match=r"strategy.simulation accepts only"):
        create_exploration(recipe)


@pytest.mark.parametrize("legacy", [False, True])
def test_snapshot_restart_preserves_exact_fingerprint(search, legacy):
    import numpy as np
    from gdpx.execution.fingerprint import structure_digest, write_structure_inputs

    obj = search()
    atoms = Atoms("Cu2", positions=[[0.123456789123, 0, 0], [2.987654321987, 0, 0]],
                  cell=[10]*3, pbc=True)
    atoms.set_tags([1, 2])
    frames, records = obj._select_snapshots(
        1, {0: [atoms]}, {"seed": 13, "parents": [0]})
    root = obj.directory / "gen1"
    assert structure_digest(read(root / "sel/selected.xyz", ":")) != structure_digest(frames)
    if legacy:
        write_structure_inputs(root / "min/_meta/structures.json", frames)
        (root / "sel/selected.json").unlink()
        (root / "sel/selection.json").write_text(json.dumps({"snapshots": records}))
    restored, restored_records = obj._select_snapshots(1, {}, {})
    assert restored_records == records
    assert structure_digest(restored) == structure_digest(frames)
    np.testing.assert_array_equal(restored[0].positions, atoms.positions)
    assert (root / "sel/selected.json").exists()
    assert obj._select_snapshots(1, {}, {})[1] == records


def test_missing_lossless_snapshot_is_not_replaced_by_rounded_xyz(search):
    obj = search()
    obj._select_snapshots(1, {0: [Atoms("Cu")]}, {"seed": 13, "parents": [0]})
    (obj.directory / "gen1/sel/selected.json").unlink()
    with pytest.raises(RuntimeError, match="requires the saved selected.json"):
        obj._select_snapshots(1, {}, {})



def test_final_report_reuses_population_but_standalone_and_resume_refresh(search, monkeypatch):
    from gdpx.exploration.population.population import Population

    calls = []
    original = Population.refresh

    def tracked(population, *args, **kwargs):
        calls.append(population)
        return original(population, *args, **kwargs)

    monkeypatch.setattr(Population, "refresh", tracked)
    obj = search()
    obj.run()
    # Generation 0, generation 1 parent selection, generation 1 completion.
    assert len(calls) == 3
    retained = [a.info["confid"] for a in obj.population.candidates]
    exported = read(obj.directory / "results/all_candidates.xyz", ":")
    energies = [a.get_potential_energy() for a in exported]

    calls.clear()
    standalone = search()
    standalone.report()
    assert len(calls) == 1
    assert [a.info["confid"] for a in standalone.population.candidates] == retained

    calls.clear()
    resumed = search()
    resumed.run()  # All generations are already complete; rebuild once for reporting.
    assert len(calls) == 1
    assert [a.info["confid"] for a in resumed.population.candidates] == retained
    assert [a.get_potential_energy() for a in read(
        obj.directory / "results/all_candidates.xyz", ":")] == energies


@pytest.mark.parametrize('provider,backend', [('ase', 'ase'), ('lammps', None)])
def test_simulation_inherits_runtime_and_resolves_backend(search, monkeypatch, provider, backend):
    obj = search()
    runtime = obj.worker.as_dict()
    runtime['potential'] = dict(provider='mace', backend='ase', parameters=dict(model='missing.model'))
    runtime['scheduler'] = dict(provider='slurm')
    original = copy.deepcopy(runtime)
    monkeypatch.setattr(obj.worker, 'as_dict', lambda: runtime)
    obj.strategy_config['simulation']['provider'] = provider
    from types import SimpleNamespace
    from gdpx.exploration.simulated_annealing import population
    captured = []
    monkeypatch.setattr(population, 'create_worker', lambda config, **kw: captured.append(config) or SimpleNamespace())
    obj._cooling(1, [], 1)
    resolved = captured[0]
    assert resolved['potential'].get('backend') == backend
    assert resolved['potential']['parameters'] == runtime['potential']['parameters']
    assert resolved['scheduler'] == runtime['scheduler']
    assert resolved['dispatch'] == runtime['dispatch']
    assert runtime == original


def test_generation_zero_does_not_construct_cooling_worker(search, monkeypatch):
    obj = search()
    config = obj.as_dict()
    runtime = config.pop('runtime')
    config['strategy']['temperatures'] = []
    config['strategy']['convergence'] = {'generation': 0}
    rebuilt = create_exploration(config)
    rebuilt.directory = obj.directory
    rebuilt.register_worker(create_worker(runtime))
    monkeypatch.setattr(rebuilt, '_cooling', lambda *args: pytest.fail('Unexpected MD worker'))
    rebuilt.run()
    assert rebuilt.read_convergence()
    assert not (rebuilt.directory / 'gen1').exists()


@pytest.mark.parametrize('tags', [None, [0, 0], [-1, 1]])
def test_sa_requires_search_tags_before_calculation(search, monkeypatch, tags):
    obj = search()
    path = obj.directory.parent / 'seed.xyz'
    atoms = read(path)
    if tags is None:
        atoms.arrays.pop('tags')
    else:
        atoms.set_tags(tags)
    write(path, atoms)
    from gdpx.exploration.simulated_annealing import population
    monkeypatch.setattr(population, 'create_worker', lambda *a, **kw: pytest.fail('Calculation started'))
    with pytest.raises(ValueError, match='tags'):
        obj.run()


def test_sa_search_space_freezes_both_stages_and_survives_restart(search):
    source = search()
    config = source.as_dict()
    runtime = config.pop('runtime')
    config['system']['search_space'] = {'active': 'index 1', 'frozen': 'index 0'}
    original = copy.deepcopy((config, runtime))

    def build():
        obj = create_exploration(copy.deepcopy(config))
        obj.directory = source.directory
        obj.register_worker(create_worker(copy.deepcopy(runtime)))
        return obj

    obj = build()
    assert obj._minimum_runtime()['executor']['parameters']['stop']['fmax'] == .1
    obj.run()
    build().run()
    assert (config, runtime) == original
    policy = obj.population_config.search_space
    for candidate in obj.get_minimised_candidates():
        assert policy.select(candidate) == [1]
        assert policy.select(candidate, 'frozen') == [0]
        assert candidate.positions[0].tolist() == [4., 5., 5.]
    saved = obj.as_dict()
    runtime = saved.pop('runtime')
    rebuilt = create_exploration(saved)
    rebuilt.register_worker(create_worker(runtime))
    assert rebuilt.as_dict() == obj.as_dict()


def test_sa_search_space_rejects_different_stage_constraints(search, monkeypatch):
    source = search()
    config = source.as_dict()
    runtime = config.pop('runtime')
    config['system']['search_space'] = {'active': 'index 1', 'frozen': 'index 0'}
    config['strategy']['simulation']['parameters']['setup']['constraint'] = 'index 1'
    obj = create_exploration(config)
    obj.directory = source.directory
    obj.register_worker(create_worker(runtime))
    monkeypatch.setattr(obj, '_evaluate_stage', lambda *a, **kw: pytest.fail('Calculation started'))
    with pytest.raises(ValueError, match='inconsistent'):
        obj.run()
