"""Reject incompatible annealing inputs before execution setup."""

import copy

import pytest

from gdpx.cli import explore
from gdpx.exploration.factory import validate_exploration_runtime
from gdpx.providers.configuration import RuntimeConfig


def runtime(provider="ase", regulator="nose_hoover_chain"):
    return {
        "potential": {"provider": "mace", "parameters": {"model": "missing.model"}},
        "executor": {"provider": provider, "method": "md", "parameters": {
            "setup": {"ensemble": {"method": "nvt", "temperature": 600},
                      "regulator": {"name": regulator}},
            "stop": {"steps": 200}}},
    }


@pytest.mark.parametrize("method", ["simulated_annealing", "global_optimisation"])
@pytest.mark.parametrize("mode", ["direct", "slurm", "spawn"])
@pytest.mark.parametrize("external", [False, True])
def test_rejects_nhc_before_any_setup(tmp_path, monkeypatch, method, mode, external):
    params = {"method": method, "strategy": {"method": "simulated_annealing"},
              "scheduler": {"provider": "slurm" if mode == "slurm" else "direct"}}
    invalid = runtime()
    # The effective override, rather than the embedded runtime, is validated.
    params["runtime"] = runtime(regulator="langevin") if external else invalid
    original = copy.deepcopy(params)

    def unexpected(*args, **kwargs):
        pytest.fail("Input validation must happen before construction or submission")

    for name in ("canonicalise_scheduler", "create_exploration", "create_worker", "ExplorationBasedWorker"):
        monkeypatch.setattr(explore, name, unexpected)
    directory = tmp_path / "not_created"
    with pytest.raises(ValueError, match="ASE nose_hoover_chain.*set_temperature"):
        explore.run_exploration(params, runtime=invalid if external else None,
                                directory=directory, spawn="0" if mode == "spawn" else None)
    assert not directory.exists()
    assert params == original


@pytest.mark.parametrize("form", ["flat", "typed", "list"])
def test_rejects_alternative_runtime_forms(form):
    value = runtime()
    if form == "flat":
        value["executor"]["parameters"] = {
            "ensemble": "nvt", "temp": 600, "steps": 200,
            "controller": {"name": "nose_hoover_chain"}}
    elif form == "typed":
        value = RuntimeConfig.from_mapping(value)
    else:
        value = [runtime(regulator="berendsen"), value]
    with pytest.raises(ValueError, match="ASE nose_hoover_chain"):
        validate_exploration_runtime({"method": "simulated_annealing"}, value)


@pytest.mark.parametrize("provider,regulator", [
    ("ase", "langevin"), ("ase", "berendsen"), ("lammps", "nose_hoover_chain"),
])
def test_accepts_supported_annealing_combinations(provider, regulator):
    validate_exploration_runtime(
        {"method": "global_optimisation", "strategy": {"method": "simulated_annealing"}},
        runtime(provider, regulator),
    )


def test_does_not_restrict_other_explorations():
    validate_exploration_runtime(
        {"method": "global_optimisation", "strategy": {"method": "basin_hopping"}}, runtime())
