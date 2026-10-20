import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from gdpx.exploration.factory import create_exploration
from gdpx.execution.factory import create_worker
from gdpx.exploration.simulated_annealing.simulated_annealing import SimulatedAnnealing


@pytest.mark.parametrize("temperatures", [[], [0.], [-1.], [float("nan")], [True]])
def test_invalid_schedule(temperatures):
    with pytest.raises(ValueError, match="temperatures"):
        SimulatedAnnealing(None, temperatures)


@pytest.mark.parametrize("provider", ["ase", "lammps", "vasp", "cp2k", "custom"])
@pytest.mark.parametrize("ensemble", ["nvt", "npt"])
def test_md_runtime_validation_is_provider_independent(monkeypatch, provider, ensemble):
    from types import SimpleNamespace
    from gdpx.exploration.exploration import BaseExploration

    # Isolate the exploration contract from optional backend installations.
    monkeypatch.setattr(BaseExploration, "register_worker",
                        lambda self, worker, *args, **kwargs: setattr(self, "worker", worker))
    worker = SimpleNamespace(
        runtime=SimpleNamespace(config=SimpleNamespace(
            executor=SimpleNamespace(provider=provider, method="md", parameters=dict(
                ensemble=ensemble, steps=10, controller=dict(name="nose_hoover"))),
            modifiers=[])),
        _drivers=[object()], _share_wdir=False)
    annealing = SimulatedAnnealing(None, annealing=dict(
        start_temperature=600., final_temperature=300.))
    annealing.register_worker(worker)
    assert annealing.worker is worker
    worker.runtime.config.executor.method = "min"
    with pytest.raises(ValueError, match="requires executor.method='md'"):
        annealing.register_worker(worker)


@pytest.mark.parametrize("fmax", [None, True, 0., -1., float("nan"), float("inf")])
def test_population_requires_valid_minimisation(fmax):
    from gdpx.exploration.simulated_annealing.population import PopulationAnnealing
    with pytest.raises(ValueError, match="positive stop.steps and stop.fmax"):
        PopulationAnnealing(None, {"method": "simulated_annealing", "temperatures": [[600, 300]],
            "minimisation": {"provider": "ase", "method": "min",
                             "parameters": {"stop": {"steps": 10, "fmax": fmax}}}})


def engine(path):
    obj = create_exploration(dict(method="simulated_annealing", recipe=dict(
        random_seed=17, temperatures=[600., 300.],
        builder=dict(method="direct", frames=[
            Atoms("Cu2", positions=[[0, 0, 0], [2.4, 0, 0]], cell=[8]*3, pbc=True),
            Atoms("Cu2", positions=[[0, 0, 0], [2.6, 0, 0]], cell=[8]*3, pbc=True)]))))
    obj.directory = path
    obj.register_worker(create_worker(dict(
        potential=dict(provider="emt"),
        executor=dict(provider="ase", method="md", parameters=dict(
            ensemble=dict(method="nvt", temperature=900.),
            setup=dict(timestep=.1,
                       velocities=dict(initialize="always", remove_rotation=False),
                       regulator=dict(name="berendsen",
                                      parameters=dict(Tdamp=10.))),
            stop=dict(steps=2), output=dict(trajectory=dict(period=1)))))))
    return obj


def test_md_cooling_continuity_and_restart(tmp_path, monkeypatch):
    obj = engine(tmp_path)
    original = obj._worker
    def interrupt(stage):
        if stage == 1:
            raise RuntimeError("interrupted")
        return original(stage)
    monkeypatch.setattr(obj, "_worker", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        obj.run()
    resumed = engine(tmp_path)
    resumed.run()
    assert resumed.read_convergence()
    trajectories = [w.retrieve(include_retrieved=True) for w in resumed.get_workers()]
    for a, b in zip(trajectories[0], trajectories[1]):
        np.testing.assert_allclose(a[-1].positions, b[0].positions)
        np.testing.assert_allclose(a[-1].get_momenta(), b[0].get_momenta())
    final = read(tmp_path / "final.xyz", ":")
    assert len(final) == 2
    assert all(np.isfinite(a.get_potential_energy()) for a in final)
    before = (tmp_path / "final.xyz").read_bytes()
    resumed.run()
    assert before == (tmp_path / "final.xyz").read_bytes()
    resumed.temperatures = [600., 200.]
    with pytest.raises(ValueError, match="configuration changed"):
        resumed.run()


@pytest.mark.parametrize("boundary", ["cycle", "ingestion"])
def test_population_cycles_resume_and_retain_minima(tmp_path, monkeypatch, boundary):
    import copy
    import json
    import yaml
    from pathlib import Path
    from gdpx.exploration.persist.database import GlobalOptimisationDatabase
    monkeypatch.chdir(Path(__file__).resolve().parents[2] / "examples/global_optimisation/explorations/simulated_annealing")
    config = yaml.safe_load(Path("cu8.yaml").read_text())
    assert "runtime" not in config
    runtime = yaml.safe_load(Path("../../runtimes/emt_sa.yaml").read_text())
    md = runtime["executor"]["parameters"]
    md["stop"]["steps"] = 1
    md["output"]["trajectory"]["period"] = 1
    md["ensemble"]["temperature"] = 600.
    config["strategy"]["minimisation"]["parameters"]["stop"] = dict(steps=100, fmax=.1)
    config["strategy"]["temperatures"] = [[600., 300.], [450., 200.]]
    config["strategy"]["convergence"]["generation"] = 2
    config["system"]["generation"]["total_size"] = 2
    def construct():
        obj = create_exploration(copy.deepcopy(config))
        obj.directory = tmp_path
        obj.register_worker(create_worker(runtime))
        return obj
    obj = construct()
    serialized = obj.as_dict()
    saved_runtime = serialized.pop("runtime")
    rebuilt = create_exploration(serialized)
    rebuilt.register_worker(create_worker(saved_runtime))
    assert rebuilt.as_dict() == obj.as_dict()
    original = obj._cooling
    def interrupt(generation, *args):
        if generation == 2 and boundary == "cycle":
            raise RuntimeError("interrupted")
        return original(generation, *args)
    monkeypatch.setattr(obj, "_cooling", interrupt)
    record = GlobalOptimisationDatabase.add_evaluated_candidate
    def interrupt_record(database, atoms, evaluation_key):
        result = record(database, atoms, evaluation_key)
        if boundary == "ingestion" and evaluation_key == "sa:1:0":
            raise RuntimeError("interrupted")
        return result
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_evaluated_candidate", interrupt_record)
    with pytest.raises(RuntimeError, match="interrupted"):
        obj.run()
    monkeypatch.setattr(GlobalOptimisationDatabase, "add_evaluated_candidate", record)
    resumed = construct()
    resumed.run()
    assert resumed.read_convergence()
    db = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    assert len(list(db.connection.select(relaxed=1))) == 6
    state = json.loads((tmp_path / "population.json").read_text())
    assert len(state["plans"]["1"]["parents"]) == 2
    assert set(state["plans"]["2"]["parents"]) <= set(state["plans"]["1"]["retained"])
    assert len(resumed.population.candidates) <= 2
    results = tmp_path / "results"
    assert not (tmp_path / "population.xyz").exists()
    assert not (results / "population.xyz").exists()
    assert len(read(results / "all_candidates.xyz", ":")) == 6
    assert (results / "pop.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(resumed.get_workers()) == 5
    for worker in resumed.get_workers()[1:]:
        definition = worker.metadata.calculation_set(worker.metadata.inputs.read(), ".")
        seeds = [seed for batch in definition["batches"] for seed in batch["random_seeds"]]
        assert len(seeds) == len(set(seeds)) == 2
    for generation in (1, 2):
        assert (tmp_path / f"gen{generation}" / "min").is_dir()
        assert (tmp_path / f"gen{generation}" / "sim").is_dir()
        assert not (tmp_path / f"gen{generation}" / "minimisation").exists()
        assert not (tmp_path / f"gen{generation}" / "cooling").exists()
        assert not (tmp_path / f"gen{generation}" / "inputs.traj").exists()
        cooling = resumed._cooling(generation, read(tmp_path / f"gen{generation}" / "inputs.xyz", ":"),
                                  state["plans"][str(generation)]["seed"])
        from gdpx.providers.configuration import resolve_executor_parameters
        params = resolve_executor_parameters(cooling.runtime.config.executor.parameters, "md")
        assert [params["temp"], params["tend"]] == [[600., 300.], [450., 200.]][generation - 1]
    (results / "pop.png").unlink()
    resumed.run()
    assert (results / "pop.png").is_file()
    assert len(list(db.connection.select(relaxed=1))) == 6
    config["strategy"]["temperatures"][0] = [500., 300.]
    with pytest.raises(ValueError, match="configuration changed"):
        construct().run()


@pytest.mark.parametrize("annealing", [
    {}, {"start_temperature": True, "final_temperature": 300.},
    {"start_temperature": 600., "final_temperature": 0.},
    {"start_temperature": float("inf"), "final_temperature": 300.},
    {"start_temperature": 300., "final_temperature": 600.},
    {"start_temperature": 600., "final_temperature": 300., "schedule": "unknown"},
])
def test_invalid_ramp(annealing):
    with pytest.raises(ValueError):
        SimulatedAnnealing(None, annealing=annealing)


def test_evolution_uses_objective_and_parent_links(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from gdpx.exploration.simulated_annealing.results import write_results

    candidates = []
    for identifier, generation, target, parents in [
        (0, 0, -2., []), (1, 1, -3., [0]), (2, 2, -1., [0]),
    ]:
        atoms = Atoms("Cu")
        atoms.info.update(confid=identifier, key_value_pairs=dict(generation=generation, target=target),
                          data=dict(parents=parents))
        candidates.append(atoms)
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, "subplots", lambda **kwargs: (fig, ax))
    write_results(tmp_path, candidates, [candidates[1]], "formation_energy")
    assert not (tmp_path / "population.xyz").exists()
    assert len(read(tmp_path / "all_candidates.xyz", ":")) == 3
    assert ax.get_ylabel() == "Formation energy [eV]"
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [[0, -2], [1, -3], [2, -1]])
    np.testing.assert_allclose(ax.collections[1].get_offsets(), [[1, -3]])
    np.testing.assert_allclose(ax.lines[0].get_xydata(), [[0, -2], [1, -3]])
    np.testing.assert_allclose(ax.lines[1].get_xydata(), [[0, -2], [2, -1]])
    np.testing.assert_allclose(ax.lines[2].get_ydata(), [-2, -3, -3])
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize("thermostat", ["berendsen", "langevin", "default"])
def test_linear_ramp_targets(tmp_path, monkeypatch, thermostat):
    from gdpx.providers.ase import driver
    from gdpx.providers.configuration import resolve_executor_parameters
    from ase import units
    obj = engine(tmp_path)
    runtime = obj.worker.as_dict()
    params = resolve_executor_parameters(runtime["executor"]["parameters"], "md")
    params["controller"] = {} if thermostat == "default" else dict(name=thermostat)
    runtime["executor"]["parameters"] = params
    ramp = SimulatedAnnealing(obj.builder, annealing=dict(start_temperature=600., final_temperature=300.),
                             directory=tmp_path, random_seed=17)
    ramp.register_worker(create_worker(runtime))
    targets = []
    update = driver.update_target_temperature
    def record(dyn, dtemp):
        update(dyn, dtemp)
        targets.append(dyn.temp / units.kB if thermostat == "langevin" else dyn.get_temperature())
    monkeypatch.setattr(driver, "update_target_temperature", record)
    ramp.run()
    np.testing.assert_allclose(targets, [600., 450., 300.] * 2)
    assert len(ramp.get_workers()) == 1
    assert all(a.info["annealing_temperature"] == 300. for a in read(tmp_path / "final.xyz", ":"))


@pytest.mark.parametrize("temperatures,generations", [
    (None, 1), ({}, 1), ([True], 1), ([0], 1), ([-1], 1),
    ([float("nan")], 1), ([float("inf")], 1), ([600], 1), ([], 1),
    ([[600, 300]], 2), ([[600]], 1), ([[600, 300, 100]], 1),
    ([[300, 600]], 1), ([[300, 300]], 1), ([[600, 0]], 1),
    ([[True, 100]], 1), ([[600, False]], 1), ([[600, -1]], 1),
    ([[float("nan"), 100]], 1), ([[600, float("inf")]], 1),
    ([[600, "300"]], 1),
])
def test_population_rejects_invalid_generation_temperatures(temperatures, generations):
    from gdpx.exploration.simulated_annealing.population import PopulationAnnealing
    strategy = {"method": "simulated_annealing", "temperatures": temperatures,
                "minimisation": {"provider": "ase", "method": "min",
                                 "parameters": {"stop": {"steps": 100, "fmax": .1}}}}
    with pytest.raises(ValueError, match="temperature"):
        PopulationAnnealing(None, strategy, convergence={"generation": generations})


def test_population_defaults_generation_count_and_preserves_surface(monkeypatch):
    from pathlib import Path
    import copy
    import yaml
    root = Path(__file__).resolve().parents[2] / "examples/global_optimisation"
    monkeypatch.chdir(root / "explorations/simulated_annealing")
    recipe = yaml.safe_load(Path("cu8.yaml").read_text())
    recipe["strategy"].pop("convergence")
    recipe["strategy"]["temperatures"] = [[600, 300], [900, 200], [600, 300]]
    runtime = yaml.safe_load((root / "runtimes/emt_sa.yaml").read_text())
    original = copy.deepcopy(runtime)
    engine = create_exploration(recipe)
    engine.register_worker(create_worker(runtime))
    assert engine.convergence == {"generation": 3}
    for i, expected in enumerate(recipe["strategy"]["temperatures"], 1):
        worker = engine._cooling(i, [], 0)
        params = worker.runtime.config.executor.parameters
        assert [params["temp"], params["tend"]] == expected
        assert worker.runtime.config.potential == engine.worker.runtime.config.potential
    minimum = engine._minimum_runtime()
    assert minimum["potential"] == engine.worker.runtime.config.potential.to_dict()
    assert minimum["executor"]["method"] == "min"
    assert runtime == original
    recipe["strategy"]["minimisation"]["potential"] = {"provider": "emt"}
    with pytest.raises(ValueError, match="Unknown executor fields"):
        create_exploration(recipe)


@pytest.mark.parametrize("stage", ["sim", "min"])
def test_population_pending_generation_resumes(tmp_path, monkeypatch, stage):
    from pathlib import Path
    import yaml
    from gdpx.execution.schedulers.direct import DirectScheduler
    from gdpx.exploration.persist.database import GlobalOptimisationDatabase
    root = Path(__file__).resolve().parents[2] / "examples/global_optimisation"
    monkeypatch.chdir(root / "explorations/simulated_annealing")
    recipe = yaml.safe_load(Path("cu8.yaml").read_text())
    recipe["strategy"]["temperatures"] = [[450, 150]]
    recipe["strategy"]["convergence"] = {"generation": 1}
    recipe["strategy"]["minimisation"]["parameters"]["stop"] = {"steps": 100, "fmax": .1}
    recipe["system"]["generation"]["total_size"] = 1
    runtime = yaml.safe_load((root / "runtimes/emt_sa.yaml").read_text())
    runtime["executor"]["parameters"]["stop"]["steps"] = 1
    runtime["executor"]["parameters"]["output"]["trajectory"]["period"] = 1
    submit, finished = DirectScheduler.submit, DirectScheduler.is_finished
    pending, seen = {}, set()

    def defer(scheduler, func_to_execute=None):
        key = str(scheduler.script)
        if f"gen1/{stage}/" in key and key not in seen:
            seen.add(key)
            pending[key] = func_to_execute
            return "pending"
        return submit(scheduler, func_to_execute)

    monkeypatch.setattr(DirectScheduler, "submit", defer)
    monkeypatch.setattr(DirectScheduler, "is_finished",
                        lambda scheduler: str(scheduler.script) not in pending and finished(scheduler))
    engine = create_exploration(recipe)
    engine.directory = tmp_path
    engine.register_worker(create_worker(runtime))
    engine.run()
    assert not engine.read_convergence()
    assert len(pending) == 1
    callbacks = list(pending.values())
    pending.clear()
    for callback in callbacks:
        callback()
    restored = create_exploration(recipe)
    restored.directory = tmp_path
    restored.register_worker(create_worker(runtime))
    restored.run()
    assert restored.read_convergence()
    assert len(seen) == 1
    database = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    assert len(list(database.connection.select(relaxed=1))) == 3


@pytest.mark.parametrize("thermostat", ["berendsen", "langevin"])
def test_population_cools_within_each_generation(tmp_path, monkeypatch, thermostat):
    from pathlib import Path
    import yaml
    from ase import units
    from gdpx.providers.ase import driver
    from gdpx.providers.configuration import resolve_executor_parameters

    root = Path(__file__).resolve().parents[2] / "examples/global_optimisation"
    monkeypatch.chdir(root / "explorations/simulated_annealing")
    recipe = yaml.safe_load(Path("cu8.yaml").read_text())
    recipe["strategy"]["temperatures"] = [[600, 300], [900, 100]]
    recipe["strategy"]["convergence"] = {"generation": 2}
    runtime = engine(tmp_path / "fixture").worker.as_dict()
    params = resolve_executor_parameters(runtime["executor"]["parameters"], "md")
    params["controller"] = dict(name=thermostat)
    runtime["executor"]["parameters"] = params
    population = create_exploration(recipe)
    population.directory = tmp_path / "population"
    population.register_worker(create_worker(runtime))
    targets = []
    update = driver.update_target_temperature

    def record(dyn, dtemp):
        update(dyn, dtemp)
        targets.append(dyn.temp / units.kB if thermostat == "langevin" else dyn.get_temperature())

    monkeypatch.setattr(driver, "update_target_temperature", record)
    for generation in (1, 2):
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.4, 0, 0]], cell=[8]*3, pbc=True)
        worker = population._cooling(generation, [atoms], 17)
        worker.run([atoms], rng_states=[17])
    np.testing.assert_allclose(targets, [600, 450, 300, 900, 500, 100])
