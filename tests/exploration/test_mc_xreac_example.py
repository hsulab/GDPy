"""Short xreac Monte Carlo example."""
import copy
from pathlib import Path

import pytest
import yaml
from ase.constraints import FixAtoms
from ase.io import read

from gdpx.cli.explore import run_exploration


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "monte_carlo" / "cu111-oxidation-xreac.yaml"
HMC_EXAMPLE = ROOT / "examples" / "monte_carlo" / "hybrid-cu111-oxidation-xreac.yaml"


def test_cu111_oxidation_accepts_oxygen_exchange(tmp_path, monkeypatch):
    pytest.importorskip("xreac")
    monkeypatch.chdir(ROOT)
    params = yaml.safe_load(EXAMPLE.read_text())
    initial = read(params["system"]["builder"]["fname"])
    assert len(initial) == 12
    assert len(initial.constraints) == 1
    assert isinstance(initial.constraints[0], FixAtoms)
    assert initial.constraints[0].get_indices().tolist() == [0, 1, 2, 3]

    run_exploration(copy.deepcopy(params), directory=tmp_path)

    frames = read(tmp_path / "mc.xyz", ":")
    assert [frame.get_chemical_formula() for frame in frames] == [
        "Cu12", "Cu12O", "Cu12O", "Cu12O",
    ]
    rows = (tmp_path / "opstat.txt").read_text().splitlines()
    assert "Insert_O_" in rows[1]
    assert "True" in rows[1]


def test_hybrid_cu111_oxidation_moves_surface_and_accepts_oxygen(tmp_path, monkeypatch):
    pytest.importorskip("xreac")
    monkeypatch.chdir(ROOT)
    params = yaml.safe_load(HMC_EXAMPLE.read_text())
    initial = read(params["system"]["builder"]["fname"])

    run_exploration(copy.deepcopy(params), directory=tmp_path)

    frames = read(tmp_path / "mc.xyz", ":")
    assert [frame.get_chemical_formula() for frame in frames] == ["Cu12", "Cu12O"]
    assert (frames[-1].positions[:4] == initial.positions[:4]).all()
    assert not (frames[-1].positions[4:12] == initial.positions[4:12]).all()
    rows = (tmp_path / "mcmoves.log").read_text().splitlines()
    assert len(rows) == 4
    assert "Insert_O_" in rows[1]
    assert "True" in rows[1]
