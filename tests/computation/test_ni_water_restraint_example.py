"""Short xreac distance-restraint example on H2O/Ni(111)."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.constraints import FixAtoms
from ase.io import read

from gdpx.execution.factory import create_worker
from gdpx.providers import expand_runtime_configs


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "compute" / "ni111_water_restraint"


def load_example():
    return yaml.safe_load((EXAMPLE / "runtime.yaml").read_text()), read(EXAMPLE / "structure.xyz")


def test_ni_water_restraint_example_configuration():
    source, atoms = load_example()
    configs = expand_runtime_configs(source)

    assert atoms.get_chemical_formula() == "H2Ni8O"
    assert atoms.pbc.tolist() == [True, True, False]
    assert len(atoms.constraints) == 1
    assert isinstance(atoms.constraints[0], FixAtoms)
    assert atoms.constraints[0].get_indices().tolist() == [0, 1, 2, 3]
    assert len(configs) == 4
    assert [config.modifiers[0].parameters["center"] for config in configs] == [
        0.95,
        1.15,
        1.35,
        1.55,
    ]
    for config in configs:
        assert config.potential.provider == "reax"
        assert config.potential.backend == "xreac"
        assert config.potential.parameters["model"] == "bundled:ffield.reax.PtNiCHO.2016"
        assert config.executor.method == "md"
        assert config.modifiers[0].method == "distance_harmonic"
        assert config.modifiers[0].parameters["group"] == "`index 8 10`"
        assert config.modifiers[0].parameters["kspring"] == 5.0


def test_ni_water_restraint_example_runs(tmp_path, monkeypatch):
    pytest.importorskip("xreac")
    monkeypatch.chdir(ROOT)
    source, atoms = load_example()
    config = expand_runtime_configs(source)[-1]
    initial_positions = atoms.positions.copy()
    initial_distance = atoms.get_distance(8, 10, mic=True)

    worker = create_worker(config, directory=tmp_path, print_func=lambda _: None)
    worker.run([atoms])
    worker.inspect([atoms])
    results = worker.retrieve(include_retrieved=True)

    assert len(results) == 1
    frames = read(tmp_path / "cand0" / "traj.xyz", ":")
    assert len(frames) == 5
    np.testing.assert_allclose(frames[-1].positions[:4], initial_positions[:4])

    log_path = tmp_path / "cand0" / "01.DistanceHarmonicCalculator" / "calc.log"
    rows = [line.split() for line in log_path.read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 21
    assert int(rows[0][0]) == 0
    assert float(rows[0][1]) == pytest.approx(initial_distance, abs=5e-5)
    expected_bias = 0.5 * 5.0 * (initial_distance - 1.55) ** 2
    assert float(rows[0][2]) == pytest.approx(expected_bias, abs=5e-4)
