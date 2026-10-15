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


@pytest.mark.parametrize("fmax", [None, True, 0., -1., float("nan"), float("inf")])
def test_population_requires_valid_minimisation(fmax):
    from gdpx.exploration.simulated_annealing.population import PopulationAnnealing
    with pytest.raises(ValueError, match="positive steps and fmax"):
        PopulationAnnealing(None, dict(method="simulated_annealing",
                                       annealing=dict(start_temperature=600., final_temperature=300.),
                                       minimisation=dict(stop=dict(steps=10, fmax=fmax))))


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
            setup=dict(ensemble="nvt", timestep=.1,
                       velocities=dict(initialize="always", remove_rotation=False),
                       regulator=dict(name="berendsen", targets=dict(temperature=900.),
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
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    config = yaml.safe_load(Path("examples/global_optimisation/explorations/simulated_annealing/cu8.yaml").read_text())
    runtime = config.pop("runtime")
    runtime["executor"]["parameters"]["stop"]["steps"] = 1
    runtime["executor"]["parameters"]["output"]["trajectory"]["period"] = 1
    config["strategy"]["annealing"] = dict(start_temperature=600., final_temperature=300.)
    config["strategy"]["convergence"]["generation"] = 2
    config["strategy"]["minimisation"]["stop"] = dict(steps=100, fmax=.1)
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
    assert len(read(tmp_path / "population.xyz", ":")) == len(resumed.population.candidates)
    assert len(resumed.get_workers()) == 5
    for generation in (1, 2):
        assert (tmp_path / f"gen{generation}" / "minimisation").is_dir()
        cooling = resumed._cooling(generation, read(tmp_path / f"gen{generation}" / "inputs.traj", ":"),
                                  state["plans"][str(generation)]["seed"])
        from gdpx.providers.configuration import resolve_executor_parameters
        params = resolve_executor_parameters(cooling.get_workers()[0].as_dict()["executor"]["parameters"], "md")
        assert params["temp"] == 600.
        assert params["tend"] == 300.
    resumed.run()
    assert len(list(db.connection.select(relaxed=1))) == 6
    config["strategy"]["annealing"]["start_temperature"] = 500.
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
