"""Exercise the documented EMT runtime through the CLI exploration entry point."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.io import read

from gdpx.cli.explore import run_exploration
from gdpx.execution.factory import create_worker
from gdpx.exploration.factory import create_exploration


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "monte_carlo"


@pytest.mark.parametrize("name", ["canonical", "semi-grand-canonical", "grand-canonical"])
def test_emt_mc_example_runs_with_scheduler_metadata(tmp_path, monkeypatch, name):
    monkeypatch.chdir(ROOT)
    recipe = yaml.safe_load((EXAMPLES / f"{name}.yaml").read_text())
    recipe["convergence"]["steps"] = 2
    runtime = yaml.safe_load((EXAMPLES / "emt.yaml").read_text())
    run_exploration(recipe, runtime=runtime, directory=tmp_path)
    assert (tmp_path / "_meta" / "_scheduler.json").is_file()
    frames = read(tmp_path / "mc.xyz", ":")
    assert len(frames) == 3
    assert all(np.isfinite(a.get_potential_energy()) for a in frames)
    assert all(np.allclose(a.cell, frames[0].cell) for a in frames)
    if name == "canonical":
        assert all(a.get_chemical_formula() == "Cu32" for a in frames)
    elif name == "semi-grand-canonical":
        assert all(len(a) == 32 and set(a.symbols) <= {"Cu", "Ni"} for a in frames)
    else:
        assert all(list(a.symbols).count("Au") == 1 for a in frames)
        assert all(np.allclose(a.positions[0], frames[0].positions[0]) for a in frames)
    # A completed CLI run can be invoked again without appending trajectory frames.
    before = (tmp_path / "mc.xyz").read_bytes()
    run_exploration(recipe, runtime=runtime, directory=tmp_path)
    assert (tmp_path / "mc.xyz").read_bytes() == before


def test_monte_carlo_requires_single_point_runtime():
    recipe = yaml.safe_load((EXAMPLES / "canonical.yaml").read_text())
    runtime = yaml.safe_load((EXAMPLES / "emt.yaml").read_text())
    engine = create_exploration(recipe)
    engine.register_worker(create_worker(runtime))

    runtime["executor"].update(method="min", parameters={"fmax": 0.05})
    with pytest.raises(ValueError, match="requires runtime.executor.method: spc"):
        engine.register_worker(create_worker(runtime))
