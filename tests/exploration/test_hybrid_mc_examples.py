"""Hybrid EMT examples and reloadable CLI inputs."""
import copy
import json
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


@pytest.mark.parametrize("name", ["canonical", "semi-grand-canonical"])
def test_hybrid_emt_cycles_and_saved_cli_input(tmp_path, monkeypatch, name):
    monkeypatch.chdir(ROOT)
    params = yaml.safe_load((EXAMPLES / f"hybrid-{name}.yaml").read_text())
    params["convergence"]["steps"] = 2
    params["num_mcmoves"] = 2
    params["extra_workers"]["md"]["executor"]["parameters"].update(steps=2, dump_period=1)
    run_exploration(copy.deepcopy(params), directory=tmp_path)
    frames = read(tmp_path / "mc.xyz", ":")
    assert len(frames) == 3
    assert all(len(a) == 32 and np.isfinite(a.get_potential_energy()) for a in frames)
    assert all(np.allclose(a.cell, frames[0].cell) for a in frames)
    assert not np.allclose(frames[0].positions, frames[-1].positions)
    if name == "canonical":
        assert all(a.get_chemical_formula() == "Cu32" for a in frames)
    else:
        assert all(set(a.symbols) <= {"Cu", "Ni"} for a in frames)
    assert len((tmp_path / "opstat.txt").read_text().splitlines()) == 5
    assert len(read(tmp_path / "mc_attempts.xyz", ":")) == 1
    for cycle in (1, 2):
        path = tmp_path / "calculations" / f"step.{cycle:04d}"
        assert (path / "procedure.0000" / "excurs" / "_meta" / "inputs.json").is_file()
        assert len(list((path / "procedure.0001").glob("proposal.*/_meta/inputs.json"))) == 2
    saved = json.loads(next((tmp_path / "_meta").glob("exp-*.json")).read_text())
    assert "runtime" in saved and "worker" not in saved
    assert saved["should_retry"] is False
    # Exercise the same path used by a saved scheduler launch script.
    before = (tmp_path / "mc.xyz").read_bytes()
    run_exploration(saved, directory=tmp_path, spawn="0")
    assert (tmp_path / "mc.xyz").read_bytes() == before


def test_hybrid_serialization_preserves_mc_settings(monkeypatch):
    monkeypatch.chdir(ROOT)
    params = yaml.safe_load((EXAMPLES / "hybrid-canonical.yaml").read_text())
    runtime = params.pop("runtime")
    params.update(ignore_atoms_tags=False, should_retry=False, restart=True)
    engine = create_exploration(params)
    engine.register_worker(create_worker(runtime))
    saved = engine.as_dict()
    saved_runtime = saved.pop("runtime")
    restored = create_exploration(saved)
    restored.register_worker(create_worker(saved_runtime))
    assert restored.ignore_atoms_tags is False
    assert restored.should_retry is False
    assert restored.restart is True
    assert restored.procedure == engine.procedure
    assert restored.extra_workers == engine.extra_workers
