"""Model-free validation of the CO/Cu(111) DPA4C example."""

import copy
from pathlib import Path

import numpy as np
import yaml
from ase.build import molecule
from ase.io import read

from gdpx.execution.factory import create_worker
from gdpx.exploration.factory import create_exploration
from gdpx.providers.configuration import resolve_executor_parameters
from gdpx.structures.groups import evaluate_constraint_expression


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/global_optimisation/explorations/simulated_annealing"


def test_co_cu111_inputs_and_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(EXAMPLES)
    config = yaml.safe_load((EXAMPLES / "co_cu111.yaml").read_text())
    runtime = yaml.safe_load((EXAMPLES / "../../runtimes/dpa4c_mini_md.yaml").read_text())
    assert "runtime" not in config
    assert runtime["potential"] == {
        "provider": "deepmd", "backend": "ase",
        "parameters": {"model": "../../../../models/dpa4c-foundation.pt2"},
    }
    engine = create_exploration(config)
    engine.directory = tmp_path
    frames = engine.population_config._prepare_initial_population(engine.builders)
    assert len(frames) == 4
    substrate = read(config["system"]["builders"]["random"]["substrates"])
    original = read(ROOT / "examples/global_optimisation/assets/cu111_p2x2_3layer.xyz")
    np.testing.assert_allclose(substrate.positions, original.positions)
    np.testing.assert_allclose(substrate.cell, original.cell)
    assert substrate.get_tags().tolist() == [0] * 12
    assert substrate.pbc.all()
    assert original.pbc.tolist() == [True, True, False]
    md = resolve_executor_parameters(runtime["executor"]["parameters"], "md")
    minimum = resolve_executor_parameters(config["strategy"]["minimisation"], "min")
    for atoms in frames:
        assert atoms.get_chemical_formula() == "CCu12O"
        assert atoms.pbc.all()
        np.testing.assert_allclose(atoms.cell.lengths(), [5.1123820279787395, 4.427452710080594, 24.])
        copper = [i for i, atom in enumerate(atoms) if atom.symbol == "Cu"]
        co = [i for i, atom in enumerate(atoms) if atom.symbol != "Cu"]
        assert len(copper) == 12 and len(co) == 2
        assert np.all(atoms.get_tags()[copper] == 0)
        assert atoms.get_tags()[co[0]] == atoms.get_tags()[co[1]] > 0
        np.testing.assert_allclose(atoms.get_distance(*co, mic=True), molecule("CO").get_distance(0, 1))
        assert 7.67424245 <= atoms[co].get_center_of_mass()[2] <= 9.67424245
        bottom = [i for i in copper if np.isclose(atoms.positions[i, 2], 2.)]
        assert len(bottom) == 4
        for params in (md, minimum):
            _, fixed = evaluate_constraint_expression(atoms, params["constraint"])
            assert sorted(fixed) == bottom
    assert md["steps"] == 1000 and md["timestep"] == .5
    assert md["controller"]["name"] == "langevin"
    assert config["strategy"]["annealing"] == {
        "start_temperature": 600., "final_temperature": 100., "schedule": "linear",
    }
    # Substitute only the potential: exercise real executor registration without
    # requiring weights or claiming DPA4C inference has been validated.
    local_runtime = copy.deepcopy(runtime)
    local_runtime["potential"] = {"provider": "emt"}
    engine.register_worker(create_worker(local_runtime))
    cooling = engine._cooling(1, frames, 73)
    worker = cooling.get_workers()[0]
    cooling_params = resolve_executor_parameters(worker.as_dict()["executor"]["parameters"], "md")
    assert cooling_params["temp"] == 600. and cooling_params["tend"] == 100.
    assert cooling_params["constraint"] == md["constraint"]
    min_params = resolve_executor_parameters(engine._minimum_runtime()["executor"]["parameters"], "min")
    assert min_params["constraint"] == minimum["constraint"]


def test_co_cu111_documented_command(tmp_path, monkeypatch):
    import sys
    from gdpx import main
    from gdpx.cli import explore

    monkeypatch.chdir(EXAMPLES)
    monkeypatch.setattr(main, "bootstrap_registries", lambda **kwargs: None)
    calls = []
    monkeypatch.setattr(explore, "run_exploration",
                        lambda params, wait, directory, runtime, **kwargs: calls.append((params, runtime)))
    monkeypatch.setattr(sys, "argv", [
        "gdp", "-d", str(tmp_path), "--log", "",
        "--runtime", "../../runtimes/dpa4c_mini_md.yaml",
        "explore", "co_cu111.yaml",
    ])
    assert main.main() is None
    assert len(calls) == 1
    assert calls[0][0]["strategy"]["method"] == "simulated_annealing"
    assert calls[0][1]["potential"]["provider"] == "deepmd"
    guide = ROOT / "docs/source/global_optimisation/examples/co-cu111.md"
    text = guide.read_text()
    for source in ("explorations/simulated_annealing/co_cu111.yaml",
                   "runtimes/dpa4c_mini_md.yaml"):
        relative = f"../../../../examples/global_optimisation/{source}"
        assert f"{{literalinclude}} {relative}" in text
        assert (guide.parent / relative).is_file()
