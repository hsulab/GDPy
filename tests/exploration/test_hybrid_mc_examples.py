"""Hybrid EMT examples and reloadable CLI inputs."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.io import read

from gdpx.cli.compute import run_computation

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "monte_carlo"


@pytest.mark.parametrize("name", ["canonical", "semi-grand-canonical"])
def test_hybrid_emt_cycles_and_saved_cli_input(tmp_path, monkeypatch, name):
    monkeypatch.chdir(ROOT)
    params = yaml.safe_load((EXAMPLES / f"hybrid-{name}.yaml").read_text())
    settings = params["executor"]["parameters"]
    settings["stop"]["steps"] = 2
    settings["setup"][1]["parameters"]["stop"]["steps"] = 2
    md = settings["setup"][0]["parameters"]
    md["stop"]["steps"] = 2
    md["output"]["trajectory"]["period"] = 1
    source = "cu32" if name == "canonical" else "cu16ni16"
    inputs = [str(EXAMPLES / "assets" / f"{source}.xyz")]
    run_computation(inputs, params, directory=tmp_path)
    chain = tmp_path / "cand0"
    frames = read(chain / "mc.xyz", ":")
    assert len(frames) == 3
    assert all(len(a) == 32 and np.isfinite(a.get_potential_energy()) for a in frames)
    assert all(np.allclose(a.cell, frames[0].cell) for a in frames)
    assert not np.allclose(frames[0].positions, frames[-1].positions)
    if name == "canonical":
        assert all(a.get_chemical_formula() == "Cu32" for a in frames)
    else:
        assert all(set(a.symbols) <= {"Cu", "Ni"} for a in frames)
    for cycle in (1, 2):
        path = chain / "calculations" / f"step.{cycle:04d}"
        assert (path / "procedure.0000" / "_meta" / "inputs.json").is_file()
        child = path / "procedure.0001" / "cand0"
        assert len((child / "opstat.txt").read_text().splitlines()) == 3
        assert len(read(child / "mc.xyz", ":")) == 3
    before = (chain / "mc.xyz").read_bytes()
    run_computation(inputs, params, directory=tmp_path)
    assert (chain / "mc.xyz").read_bytes() == before
