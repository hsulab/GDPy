"""Sampling runtimes use the compute configuration and executor contract."""
import copy

import pytest

from gdpx.execution import resolve_runtime
from gdpx.execution.factory import create_workers
from gdpx.execution.executor import SingleStructureExecutor
from gdpx.exploration.factory import create_exploration
from gdpx.providers.configuration import expand_runtime_configs


def sampling_config(method="monte_carlo"):
    mc = {"provider": "native", "method": "monte_carlo", "parameters": {
        "ensemble": {"method": "nvt", "temperature": 900.0},
        "setup": {"operators": [{"method": "move", "particles": ["Cu"], "max_disp": 0.1}]},
        "stop": {"steps": 2}}}
    executor = copy.deepcopy(mc)
    if method == "hybrid_monte_carlo":
        ensemble = mc["parameters"].pop("ensemble")
        executor = {"provider": "native", "method": method, "parameters": {
            "ensemble": ensemble,
            "setup": [{"provider": "ase", "method": "md", "parameters": {
                "random_seed": 17, "setup": {}, "stop": {"steps": 2}}}, mc],
            "stop": {"steps": 2}}}
    executor["parameters"]["random_seed"] = 7
    return {"potential": {"provider": "emt"}, "executor": executor}


def mc_parameters(config):
    params = config["executor"]["parameters"]
    return params["setup"][1]["parameters"] if isinstance(params["setup"], list) else params


@pytest.mark.parametrize("method", ["monte_carlo", "hybrid_monte_carlo"])
def test_roundtrip_and_independent_evaluation_scheduler(method):
    config = sampling_config(method)
    config["scheduler"] = {"provider": "slurm"}
    setup = mc_parameters(config)["setup"]
    setup["ignore_atoms_tags"] = False
    original = copy.deepcopy(config)
    runtime = resolve_runtime(config)
    engine = runtime.executor
    assert config == original
    assert isinstance(engine, SingleStructureExecutor)
    assert runtime.materialization is None
    assert runtime.scheduler.name == "slurm"
    mc = engine.children[1].executor if method == "hybrid_monte_carlo" else engine
    assert mc.worker.scheduler.name == "direct"
    assert mc.ignore_atoms_tags is False
    restored = resolve_runtime(engine.as_dict()).executor
    assert restored.as_dict() == engine.as_dict()
    assert restored.convergence == engine.convergence
    setup["executor"] = {"scheduler": {"provider": "pbs"}}
    engine = resolve_runtime(config).executor
    mc = engine.children[1].executor if method == "hybrid_monte_carlo" else engine
    assert mc.worker.scheduler.name == "pbs"


@pytest.mark.parametrize("method", ["monte_carlo", "hybrid_monte_carlo"])
def test_explore_rejects_sampling(method):
    with pytest.raises(ValueError, match="moved to gdp compute"):
        create_exploration({"method": method})


@pytest.mark.parametrize("method", ["monte_carlo", "hybrid_monte_carlo"])
def test_explore_cli_reports_migration_without_runtime(tmp_path, method):
    from gdpx.cli.explore import run_exploration

    with pytest.raises(ValueError, match="moved to gdp compute"):
        run_exploration({"method": method}, directory=tmp_path)


@pytest.mark.parametrize("method", ["monte_carlo", "hybrid_monte_carlo"])
@pytest.mark.parametrize("stop", [None, {}, {"steps": True}, {"steps": -1},
                                      {"steps": 1.5}, {"unknown": 1}])
def test_stop_validation_and_default(method, stop):
    config = sampling_config(method)
    config["executor"]["parameters"]["stop"] = stop
    if stop == {}:
        assert resolve_runtime(config).executor.convergence == {"steps": 1}
    else:
        with pytest.raises((ValueError, TypeError), match="stop"):
            resolve_runtime(config)
    config["executor"]["parameters"].pop("stop")
    assert resolve_runtime(config).executor.convergence == {"steps": 1}


@pytest.mark.parametrize("setup", [[], {}, [{"provider": "ase", "method": "md"}],
    [{"provider": "ase", "method": "spc"}],
    [{"provider": "native", "method": "hybrid_monte_carlo", "parameters": {"setup": []}}]])
def test_invalid_hybrid_cycle(setup):
    config = sampling_config("hybrid_monte_carlo")
    config["executor"]["parameters"]["setup"] = setup
    with pytest.raises((ValueError, TypeError)):
        resolve_runtime(config)


def test_ensemble_resolution_and_missing_chemical_potentials():
    config = sampling_config()
    setup = config["executor"]["parameters"]["setup"]
    params = config["executor"]["parameters"]
    params["ensemble"] = {"method": "suvt", "temperature": 600,
                         "chemical_potentials": {"Cu": 0.1, "Ni": 0.4}}
    setup["operators"] = [{"method": "swap_type", "particles": ["Cu", "Ni"]}]
    original = copy.deepcopy(config)
    op = resolve_runtime(config).executor.operators[0]
    assert op.temperature == 600 and op.chempots == [0.1, 0.4]
    assert op.skip_distance_check
    assert config == original
    del params["ensemble"]["chemical_potentials"]["Ni"]
    with pytest.raises(ValueError, match="Missing chemical potentials"):
        resolve_runtime(config)


def test_broadcast_keeps_sampling_variants_separate():
    config = sampling_config()
    config["executor"]["broadcast"] = {"ensemble.temperature": [300, 600]}
    workers = create_workers(expand_runtime_configs(config))
    assert len(workers) == 2
    assert [w.driver.operators[0].temperature for w in workers] == [300, 600]
    assert workers[0].driver.worker is not workers[1].driver.worker
    config["dispatch"] = {"share_workdir": True}
    with pytest.raises(ValueError, match="share_workdir"):
        create_workers(expand_runtime_configs(config))


@pytest.mark.parametrize("method", ["nvt", "uvt", "suvt"])
def test_hmc_inherits_one_ensemble_and_roundtrips(method):
    config = sampling_config("hybrid_monte_carlo")
    ensemble = config["executor"]["parameters"]["ensemble"]
    ensemble["method"] = method
    if method != "nvt":
        ensemble["chemical_potentials"] = {"Cu": -0.2, "Ni": 0.1}
        mc_parameters(config)["setup"]["operators"] = [{"method": "swap_type", "particles": ["Cu", "Ni"]}]
    original = copy.deepcopy(config)
    runtime = resolve_runtime(config)
    md, mc = runtime.executor.children
    assert md.executor.setting.ensemble == "nvt"
    assert md.executor.setting.temp == ensemble["temperature"]
    assert mc.executor.operators[0].temperature == ensemble["temperature"]
    if method != "nvt":
        assert mc.executor.operators[0].chempots == [-0.2, 0.1]
        assert "chemical_potentials" not in md.config.executor.parameters["ensemble"]
    saved = runtime.executor.as_dict()
    assert saved["executor"]["parameters"]["ensemble"] == ensemble
    assert all("ensemble" not in child["parameters"] for child in saved["executor"]["parameters"]["setup"])
    assert resolve_runtime(saved).executor.as_dict() == saved
    assert config == original


@pytest.mark.parametrize("index", [0, 1])
def test_hmc_rejects_child_ensemble(index):
    config = sampling_config("hybrid_monte_carlo")
    config["executor"]["parameters"]["setup"][index]["parameters"]["ensemble"] = {"method": "nvt", "temperature": 700}
    with pytest.raises(ValueError, match="only on the parent"):
        resolve_runtime(config)


@pytest.mark.parametrize("method", ["monte_carlo", "hybrid_monte_carlo"])
@pytest.mark.parametrize("ensemble", [{"method": "nve"}, {"method": "npt", "temperature": 300, "pressure": 1}])
def test_sampling_rejects_unsupported_ensembles(method, ensemble):
    config = sampling_config(method)
    config["executor"]["parameters"]["ensemble"] = ensemble
    with pytest.raises(ValueError, match="does not support"):
        resolve_runtime(config)


def test_hmc_rejects_child_temperature_ramp():
    config = sampling_config("hybrid_monte_carlo")
    config["executor"]["parameters"]["setup"][0]["parameters"]["setup"]["tend"] = 200
    with pytest.raises(ValueError, match="ramps"):
        resolve_runtime(config)
