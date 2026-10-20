"""Exercise the documented EMT runtime through the CLI compute entry point."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.io import read

from gdpx.cli.compute import run_computation


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "monte_carlo"


@pytest.mark.parametrize("name", ["canonical", "semi-grand-canonical", "grand-canonical"])
def test_emt_mc_example_runs_with_scheduler_metadata(tmp_path, monkeypatch, name):
    monkeypatch.chdir(ROOT)
    recipe = yaml.safe_load((EXAMPLES / f"{name}.yaml").read_text())
    recipe["executor"]["parameters"]["stop"]["steps"] = 2
    source = {"canonical": "cu32", "semi-grand-canonical": "cu16ni16", "grand-canonical": "aucu8"}[name]
    inputs = [str(EXAMPLES / "assets" / f"{source}.xyz")]
    run_computation(inputs, recipe, directory=tmp_path)
    assert (tmp_path / "_meta" / "inputs.json").is_file()
    chain = tmp_path / "cand0"
    frames = read(chain / "mc.xyz", ":")
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
    before = (chain / "mc.xyz").read_bytes()
    run_computation(inputs, recipe, directory=tmp_path)
    assert (chain / "mc.xyz").read_bytes() == before
