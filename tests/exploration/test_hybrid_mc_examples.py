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
    params["strategy"]["steps"] = 2
    params["strategy"]["cycle"][1]["steps"] = 2
    params["strategy"]["cycle"][0]["runtime"]["executor"]["parameters"].update(
        steps=2, dump_period=1
    )
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
    assert len((tmp_path / "mcmoves.log").read_text().splitlines()) == 5
    assert not (tmp_path / "opstat.txt").exists()
    assert len(read(tmp_path / "mc_attempts.xyz", ":")) == 1
    for cycle in (1, 2):
        path = tmp_path / "calculations" / f"step.{cycle:04d}"
        assert (path / "procedure.0000" / "excurs" / "_meta" / "inputs.json").is_file()
        assert len(list((path / "procedure.0001").glob("proposal.*/_meta/inputs.json"))) == 2
    saved = json.loads(next((tmp_path / "_meta").glob("exp-*.json")).read_text())
    assert "runtime" in saved and "worker" not in saved
    assert set(saved) == {"method", "random_seed", "system", "strategy", "runtime"}
    # Exercise the same path used by a saved scheduler launch script.
    before = (tmp_path / "mc.xyz").read_bytes()
    run_exploration(saved, directory=tmp_path, spawn="0")
    assert (tmp_path / "mc.xyz").read_bytes() == before


def test_hybrid_serialization_preserves_mc_settings(monkeypatch):
    monkeypatch.chdir(ROOT)
    params = yaml.safe_load((EXAMPLES / "hybrid-canonical.yaml").read_text())
    runtime = copy.deepcopy(params.pop("runtime"))
    params["system"]["ignore_atoms_tags"] = False
    engine = create_exploration(params)
    engine.register_worker(create_worker(runtime))
    saved = engine.as_dict()
    saved_runtime = saved.pop("runtime")
    restored = create_exploration(saved)
    restored.register_worker(create_worker(saved_runtime))
    assert restored.ignore_atoms_tags is False
    assert restored.should_retry is False
    assert restored.restart is False
    assert restored.cycle == engine.cycle
    assert restored.system_config == engine.system_config
    assert restored.strategy_config == engine.strategy_config


def test_hybrid_requires_single_point_initial_runtime(monkeypatch):
    monkeypatch.chdir(ROOT)
    params = yaml.safe_load((EXAMPLES / "hybrid-canonical.yaml").read_text())
    runtime = copy.deepcopy(params.pop("runtime"))
    runtime["executor"].update(method="min", parameters={"fmax": 0.05})
    engine = create_exploration(params)
    with pytest.raises(ValueError, match="hybrid_monte_carlo requires runtime.executor.method: spc"):
        engine.register_worker(create_worker(runtime))


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda c: c["strategy"].pop("cycle"), "nonempty strategy.cycle"),
        (lambda c: c["strategy"]["cycle"][0].update(method="unknown"), "molecular_dynamics or monte_carlo"),
        (lambda c: c["strategy"]["cycle"][0]["runtime"]["executor"].update(method="spc"), "requires runtime.executor.method: md"),
        (lambda c: c["strategy"]["cycle"][1]["runtime"]["executor"].update(method="md"), "requires runtime.executor.method: spc"),
        (lambda c: c["strategy"]["cycle"][1].update(steps=-1), "cycle.1.steps"),
        (lambda c: c["strategy"].update(ckpt_period=0), "strategy.ckpt_period"),
    ],
)
def test_hybrid_rejects_invalid_structured_config(monkeypatch, change, message):
    monkeypatch.chdir(ROOT)
    params = yaml.safe_load((EXAMPLES / "hybrid-canonical.yaml").read_text())
    params.pop("runtime")
    change(params)
    with pytest.raises((TypeError, ValueError), match=message):
        create_exploration(params)


def test_hybrid_custom_ensemble_keeps_operator_temperature(monkeypatch):
    monkeypatch.chdir(ROOT)
    params = yaml.safe_load((EXAMPLES / "hybrid-canonical.yaml").read_text())
    params.pop("runtime")
    params["system"]["ensemble"] = {"method": "custom"}
    params["strategy"]["operators"][0]["temperature"] = 750.0
    engine = create_exploration(params)
    assert engine.operators[0].temperature == 750.0


def test_hybrid_rejects_flat_legacy_config():
    with pytest.raises(ValueError, match="inline strategy.cycle stages"):
        create_exploration({
            "method": "hybrid_monte_carlo",
            "builder": {},
            "operators": [],
            "procedure": [],
            "num_mcmoves": 2,
            "extra_workers": {},
        })
