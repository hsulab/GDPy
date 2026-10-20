"""Sampling runtimes use the compute configuration and executor contract."""
import copy

import pytest

from gdpx.execution import resolve_runtime
from gdpx.execution.factory import create_workers
from gdpx.execution.executor import SingleStructureExecutor
from gdpx.exploration.factory import create_exploration
from gdpx.providers.configuration import expand_runtime_configs


def sampling_config(method="monte_carlo"):
    spc = {"potential": {"provider": "emt"},
           "executor": {"provider": "ase", "method": "spc"}}
    setup = {
        "ensemble": {"method": "canonical", "temperature": 900.0},
        "operators": [{"method": "move", "particles": ["Cu"], "max_disp": 0.1}],
        "evaluator": {"executor": copy.deepcopy(spc["executor"])},
    }
    if method == "hybrid_monte_carlo":
        md = copy.deepcopy(spc)
        md["executor"] = {
            "provider": "ase", "method": "md",
            "parameters": {"random_seed": 17, "setup": {"ensemble": "nve"},
                           "stop": {"steps": 2}},
        }
        setup["cycle"] = [
            {"method": "molecular_dynamics", "runtime": md},
            {"method": "monte_carlo", "steps": 2, "runtime": spc},
        ]
    return {
        "potential": {"provider": "emt"},
        "executor": {
            "provider": "gdpx", "method": method,
            "parameters": {"random_seed": 7, "setup": setup, "stop": {"steps": 2}},
        },
    }


@pytest.mark.parametrize("method", ["monte_carlo", "hybrid_monte_carlo"])
def test_roundtrip_and_independent_evaluation_scheduler(method):
    config = sampling_config(method)
    config["scheduler"] = {"provider": "slurm"}
    config["executor"]["parameters"]["setup"]["ignore_atoms_tags"] = False
    original = copy.deepcopy(config)
    runtime = resolve_runtime(config)
    engine = runtime.executor
    assert config == original
    assert isinstance(engine, SingleStructureExecutor)
    assert runtime.materialization is None
    assert not hasattr(engine, "calc")
    assert runtime.scheduler.name == "slurm"
    assert engine.worker.scheduler.name == "direct"
    assert engine.worker.runtime.config.executor.method == "spc"
    assert engine.ignore_atoms_tags is False
    assert engine.should_retry is False
    restored = resolve_runtime(engine.as_dict()).executor
    assert restored.as_dict() == engine.as_dict()
    assert restored.convergence == engine.convergence
    assert restored.operators[0].temperature == 900
    if method == "hybrid_monte_carlo":
        assert restored.cycle == engine.cycle
    config["executor"]["parameters"]["setup"]["evaluator"]["scheduler"] = {"provider": "pbs"}
    assert resolve_runtime(config).executor.worker.scheduler.name == "pbs"


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


@pytest.mark.parametrize("method", ["monte_carlo", "hybrid_monte_carlo"])
@pytest.mark.parametrize("change,message", [
    (lambda p: p.update(steps=3), "flat and structured"),
    (lambda p: p["setup"].update(steps=3), "Duplicate executor parameter"),
    (lambda p: p["setup"].pop("ensemble"), "requires setup.ensemble"),
    (lambda p: p["setup"]["ensemble"].update(method="bad"), "ensemble.method"),
    (lambda p: p["setup"]["operators"][0].update(temperature=1), "Move operator temperature"),
    (lambda p: p["setup"]["operators"][0].update(skip_distance_check=False), "detailed balance"),
    (lambda p: p["setup"]["evaluator"]["executor"].update(method="min"), "method: spc"),
    (lambda p: p["setup"].update(evaluator=None), "setup.evaluator"),
    (lambda p: p.update(output={"checkpoint": {"period": 0}}), "checkpoint.period"),
    (lambda p: p.update(output={"trajectory": {"period": True}}), "trajectory.period"),
    (lambda p: p.update(output={"checkpoint": {"keep": 0}}), "checkpoint.keep"),
])
def test_invalid_sampling_parameters(method, change, message):
    config = sampling_config(method)
    change(config["executor"]["parameters"])
    with pytest.raises((ValueError, TypeError), match=message):
        resolve_runtime(config)


@pytest.mark.parametrize("change,message", [
    (lambda s: s.pop("cycle"), "nonempty setup.cycle"),
    (lambda s: s["cycle"][0].update(method="unknown"), "molecular_dynamics or monte_carlo"),
    (lambda s: s["cycle"][0]["runtime"]["executor"].update(method="spc"), "method: md"),
    (lambda s: s["cycle"][1]["runtime"]["executor"].update(method="md"), "method: spc"),
    (lambda s: s["cycle"][1].update(steps=-1), "cycle.1.steps"),
])
def test_invalid_hybrid_cycle(change, message):
    config = sampling_config("hybrid_monte_carlo")
    change(config["executor"]["parameters"]["setup"])
    with pytest.raises((ValueError, TypeError), match=message):
        resolve_runtime(config)


def test_ensemble_resolution_custom_and_missing_chemical_potentials():
    config = sampling_config()
    setup = config["executor"]["parameters"]["setup"]
    setup["ensemble"] = {"method": "semi_grand_canonical", "temperature": 600,
                         "chemical_potentials": {"Cu": 0.1, "Ni": 0.4}}
    setup["operators"] = [{"method": "swap_type", "particles": ["Cu", "Ni"]}]
    original = copy.deepcopy(config)
    op = resolve_runtime(config).executor.operators[0]
    assert op.temperature == 600 and op.chempots == [0.1, 0.4]
    assert op.skip_distance_check
    assert config == original
    del setup["ensemble"]["chemical_potentials"]["Ni"]
    with pytest.raises(ValueError, match="Missing chemical potentials"):
        resolve_runtime(config)
    setup["ensemble"] = {"method": "custom"}
    setup["operators"][0].update(temperature=700, chempots=[0.1, 0.4])
    op = resolve_runtime(config).executor.operators[0]
    assert op.temperature == 700 and not op.skip_distance_check
    setup["ensemble"]["temperature"] = 700
    with pytest.raises(ValueError, match="custom ensemble accepts only method"):
        resolve_runtime(config)


def test_broadcast_keeps_sampling_variants_separate():
    config = sampling_config()
    config["executor"]["broadcast"] = {"setup.ensemble.temperature": [300, 600]}
    workers = create_workers(expand_runtime_configs(config))
    assert len(workers) == 2
    assert [w.driver.operators[0].temperature for w in workers] == [300, 600]
    assert workers[0].driver.worker is not workers[1].driver.worker
    config["dispatch"] = {"share_workdir": True}
    with pytest.raises(ValueError, match="share_workdir"):
        create_workers(expand_runtime_configs(config))
