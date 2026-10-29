import json

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from gdpx.providers import get_provider_manager
from gdpx.providers.ase.vibrations import AseFrequencyDriver, AseFrequencySetting


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.evaluated_positions = []

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.evaluated_positions.append(atoms.positions.copy())
        self.results = {"energy": float(np.sum(atoms.positions ** 2)),
                        "forces": -2 * atoms.positions}


@pytest.mark.parametrize("nfree", [2, 4])
def test_selected_group_spectrum_and_environment(tmp_path, nfree):
    atoms = Atoms("CuZnOH", positions=np.arange(12).reshape(4, 3) * 0.1)
    calc = HarmonicCalculator()
    driver = AseFrequencyDriver(calc, {"group": "`symbol Zn O H`", "nfree": nfree, "maxstep": 0.025},
                                directory=tmp_path)
    driver.run(atoms)
    result = json.loads((tmp_path / "vibrations.json").read_text())
    assert result["indices"] == [1, 2, 3]
    assert result["maxstep_angstrom"] == 0.025
    displacements = np.asarray(calc.evaluated_positions) - atoms.positions
    np.testing.assert_allclose(np.max(np.abs(displacements)), 0.025 * (nfree // 2))
    np.testing.assert_allclose(result["hessian_eV_angstrom2"], 2 * np.eye(9), atol=1e-12)
    expected = units._hbar * 1e10 / np.sqrt(units._e * units._amu) * np.sqrt(
        np.repeat(2 / atoms.get_masses()[1:], 3))
    np.testing.assert_allclose(result["energies_eV"], sorted(expected))
    assert np.asarray(result["modes"]).shape == (9, 4, 3)
    assert np.all(np.asarray(result["modes"])[:, 0] == 0)
    assert all(np.array_equal(p[0], atoms.positions[0]) for p in calc.evaluated_positions)
    assert driver.read_convergence()
    np.testing.assert_allclose(driver.read_trajectory()[0].positions, atoms.positions)
    # A changed geometry never reuses displacement forces from the old cache.
    calc.evaluated_positions.clear()
    atoms.positions += 0.05
    driver.run(atoms)
    assert len(calc.evaluated_positions) >= 9 * nfree + 1


def test_default_excludes_fixed_atoms_and_explicit_overlap_fails(tmp_path):
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.set_constraint(FixAtoms(indices=[0]))
    driver = AseFrequencyDriver(HarmonicCalculator(), {}, directory=tmp_path)
    driver.run(atoms)
    assert json.loads((tmp_path / "vibrations.json").read_text())["indices"] == [1]
    driver = AseFrequencyDriver(HarmonicCalculator(), {"indices": [0]}, directory=tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        driver.run(atoms)


@pytest.mark.parametrize("params", [{"maxstep": 0}, {"maxstep": float("nan")},
                                       {"nfree": 3}, {"group": "x", "indices": [0]}])
def test_invalid_settings(params):
    with pytest.raises(ValueError):
        AseFrequencySetting(**params)


@pytest.mark.parametrize("indices", [[], [-1], [1], [0, 0], [0.5]])
def test_invalid_indices(tmp_path, indices):
    driver = AseFrequencyDriver(HarmonicCalculator(), {"indices": indices}, directory=tmp_path)
    with pytest.raises(ValueError):
        driver.run(Atoms("Cu"))


def test_freq_runtime_factory(tmp_path):
    runtime = get_provider_manager().resolve_runtime({
        "potential": {"provider": "emt"},
        "executor": {"provider": "ase", "method": "freq", "parameters": {
            "setup": {"group": "`index 1`", "maxstep": 0.02}}},
    })
    driver = runtime.executor
    assert isinstance(driver, AseFrequencyDriver)
    driver.directory = tmp_path
    driver.run(Atoms("Cu2", positions=[[0, 0, 0], [2.3, 0, 0]]))
    assert driver.read_convergence()


def test_failed_rerun_is_not_complete(tmp_path):
    calc = HarmonicCalculator()
    driver = AseFrequencyDriver(calc, {}, directory=tmp_path)
    driver.run(Atoms("Cu"))

    def fail(*args, **kwargs):
        raise RuntimeError("force evaluation failed")

    calc.calculate = fail
    with pytest.raises(RuntimeError, match="force evaluation failed"):
        driver.run(Atoms("Cu"))
    assert not driver.read_convergence()


def test_imaginary_modes_are_preserved(tmp_path):
    calc = HarmonicCalculator()
    calculate = calc.calculate

    def unstable(*args, **kwargs):
        calculate(*args, **kwargs)
        calc.results["energy"] *= -1
        calc.results["forces"] *= -1

    calc.calculate = unstable
    driver = AseFrequencyDriver(calc, {}, directory=tmp_path)
    driver.run(Atoms("Cu"))
    result = json.loads((tmp_path / "vibrations.json").read_text())
    assert result["frequencies_cm1"] == [0, 0, 0]
    assert np.all(np.asarray(result["imaginary_frequencies_cm1"]) > 0)
    assert np.all(np.asarray(result["imaginary_energies_eV"]) > 0)
