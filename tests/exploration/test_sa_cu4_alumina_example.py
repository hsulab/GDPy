"""Model-free validation of the short Cu4/alumina DPA4C-mini example."""
import copy
from pathlib import Path

import numpy as np
import yaml
from ase.io import read

from gdpx.analysis.selectors import create_selector
from gdpx.data.array import AtomsNDArray
from gdpx.execution.factory import create_worker
from gdpx.exploration.factory import create_exploration
from gdpx.providers.configuration import resolve_executor_parameters
from gdpx.structures.groups import evaluate_constraint_expression


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/global_optimisation/explorations/simulated_annealing"


def test_cu4_alumina_sa_example(tmp_path, monkeypatch):
    monkeypatch.chdir(EXAMPLES)
    recipe = yaml.safe_load((EXAMPLES / "cu4_alumina111.yaml").read_text())
    runtime = yaml.safe_load((EXAMPLES / "../../runtimes/dpa4c_mini_cu4_alumina_sa.yaml").read_text())
    assert runtime["potential"] == {
        "provider": "deepmd", "backend": "ase",
        "parameters": {"model": "../../../../models/dpa4c-mini.pt2"},
    }
    engine = create_exploration(recipe)
    engine.directory = tmp_path
    frames = engine.population_config._prepare_initial_population(engine.builders)
    assert len(frames) == 4
    assert engine.population_config.gen_size == 4
    assert engine.convergence == {"generation": 2}
    substrate = read(recipe["system"]["builders"]["init"]["substrates"])
    md = resolve_executor_parameters(runtime["executor"]["parameters"], "md")
    minimum = resolve_executor_parameters(recipe["strategy"]["minimisation"]["runtime"]["executor"]["parameters"], "min")
    for atoms in frames:
        assert atoms.get_chemical_formula() == "Al72Cu4O108"
        assert atoms.pbc.all()
        np.testing.assert_allclose(atoms.cell, substrate.cell)
        np.testing.assert_allclose(atoms.positions[:180], substrate.positions)
        assert np.all(atoms.get_tags()[:180] == 0)
        assert len(set(atoms.get_tags()[180:])) == 4
        assert np.all(atoms.get_tags()[180:] > 0)
        distances = np.linalg.norm(atoms.positions[180:] - [7.1385, 8.2428, 10.0], axis=1)
        assert np.all(distances <= 1.8)
        fixed_sets = [evaluate_constraint_expression(atoms, p["constraint"])[1] for p in (md, minimum)]
        assert len(fixed_sets[0]) == 60
        assert set(fixed_sets[0]) == set(fixed_sets[1])
        assert all(i < 180 for i in fixed_sets[0])
    assert md["steps"] == 200 and md["timestep"] == .5
    assert minimum["steps"] == 20 and minimum["fmax"] == .1
    # Exercise registration and generated cooling runtimes without loading weights.
    local_runtime = copy.deepcopy(runtime)
    local_runtime["potential"] = {"provider": "emt"}
    engine.register_worker(create_worker(local_runtime))
    # Default unbounded insertion distances must remain serialisable.
    assert engine.builders["init"].as_dict()["molecular_distances"] == [None, None]
    engine._checkpoint()
    for generation, temperatures in enumerate(((600., 100.), (450., 100.)), start=1):
        worker = engine._cooling(generation, frames, 53)
        params = resolve_executor_parameters(worker.as_dict()["executor"]["parameters"], "md")
        assert (params["temp"], params["tend"]) == temperatures
        assert params["constraint"] == md["constraint"]
    selector = create_selector(recipe["strategy"]["selection"])
    selector.directory = tmp_path / "selection"
    data = AtomsNDArray([[a.copy(), a.copy()] for a in frames])
    selector.select(data)
    assert np.argwhere(data.markers).tolist() == [[i, 1] for i in range(4)]
