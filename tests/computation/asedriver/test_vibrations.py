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
    driver.setting.resume = False
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


@pytest.mark.parametrize("fail_calculation", [False, True])
def test_cleanup_failure_stops_other_calculators_and_preserves_run_error(tmp_path, fail_calculation):
    from gdpx.providers.ase.backend import EnhancedCalculator

    events = []

    class InteractiveHarmonic(HarmonicCalculator):
        def calculate(self, *args, **kwargs):
            if fail_calculation:
                raise ValueError("force evaluation failed")
            super().calculate(*args, **kwargs)

        def finalize(self):
            events.append("host")
            raise RuntimeError("host cleanup failed")

    class Modifier(HarmonicCalculator):
        def finalize(self):
            events.append("modifier")

    host, modifier = InteractiveHarmonic(), Modifier()
    # Nested mixers with a shared calculator must finalize each leaf once.
    calc = EnhancedCalculator([EnhancedCalculator([host, modifier]), modifier])
    driver = AseFrequencyDriver(calc, {}, directory=tmp_path)
    error = ValueError if fail_calculation else RuntimeError
    message = "force evaluation failed" if fail_calculation else "host cleanup failed"
    with pytest.raises(error, match=message):
        driver.run(Atoms("Cu"))
    assert events == ["host", "modifier"]


def test_unwrapped_calculator_is_finalized(tmp_path):
    events = []

    class InteractiveHarmonic(HarmonicCalculator):
        def finalize(self):
            events.append("finalize")

    driver = AseFrequencyDriver(InteractiveHarmonic(), {}, directory=tmp_path)
    driver.run(Atoms("Cu"))
    assert events == ["finalize"]


class InterruptibleHarmonic(HarmonicCalculator):
    def __init__(self, fail_after=None):
        super().__init__()
        self.fail_after = fail_after

    def calculate(self, *args, **kwargs):
        if self.fail_after is not None and len(self.evaluated_positions) >= self.fail_after:
            raise RuntimeError("interrupted")
        super().calculate(*args, **kwargs)


def test_interrupted_frequency_resumes_valid_forces(tmp_path):
    from ase.io.jsonio import decode
    atoms = Atoms("Cu")
    calc = InterruptibleHarmonic(fail_after=3)
    driver = AseFrequencyDriver(calc, {}, directory=tmp_path)
    with pytest.raises(RuntimeError, match="interrupted"):
        driver.run(atoms)
    assert not driver.read_convergence()
    valid = {}
    for path in (tmp_path / "vibrations").glob("cache.*.json"):
        if path.stat().st_size:
            valid[path.name] = path.read_bytes()
    assert len(valid) == 3
    # A new process/calculator must retain the same cache identity.
    calc = InterruptibleHarmonic()
    driver = AseFrequencyDriver(calc, {}, directory=tmp_path)
    driver.run(atoms)
    assert driver.read_convergence()
    # Four missing force calls, plus the reference trajectory evaluation.
    assert len(calc.evaluated_positions) == 5
    for name, content in valid.items():
        assert (tmp_path / "vibrations" / name).read_bytes() == content
    for path in (tmp_path / "vibrations").glob("cache.*.json"):
        assert np.asarray(decode(path.read_text())["forces"]).shape == (1, 3)
    np.testing.assert_allclose(json.loads((tmp_path / "vibrations.json").read_text())[
        "hessian_eV_angstrom2"], 2 * np.eye(3))


@pytest.mark.parametrize("content", ["", "{", "null", "{}", '{"forces": [[1, 2]]}',
                                     '{"forces": [[NaN, 0, 0]]}'])
def test_invalid_force_cache_is_recomputed(tmp_path, content):
    atoms = Atoms("Cu")
    driver = AseFrequencyDriver(HarmonicCalculator(), {}, directory=tmp_path)
    driver.run(atoms)
    (tmp_path / "vibrations/cache.0x-.json").write_text(content)
    calc = HarmonicCalculator()
    driver = AseFrequencyDriver(calc, {}, directory=tmp_path)
    driver.run(atoms)
    assert len(calc.evaluated_positions) == 2
    assert driver.read_convergence()


@pytest.mark.parametrize("change", ["delta", "nfree", "indices", "calculator", "cell",
                                   "constraints", "masses", "mixer"])
def test_changed_identity_recomputes_all_forces(tmp_path, change):
    from gdpx.providers.ase.backend import EnhancedCalculator
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]], cell=[8, 8, 8])
    calc = HarmonicCalculator()
    params = {"indices": [0]}
    driver = AseFrequencyDriver(calc, params, directory=tmp_path)
    driver.run(atoms)
    calc = HarmonicCalculator()
    if change == "delta":
        params["maxstep"] = 0.02
    elif change == "nfree":
        params["nfree"] = 4
    elif change == "indices":
        params["indices"] = [1]
    elif change == "calculator":
        calc.parameters["test_setting"] = 1
    elif change == "cell":
        atoms.cell[0, 0] += 1
    elif change == "constraints":
        atoms.set_constraint(FixAtoms(indices=[1]))
    elif change == "masses":
        atoms.set_masses([64, 64])
    elif change == "mixer":
        calc = EnhancedCalculator([calc, HarmonicCalculator()], weights=[1, 0.5])
    driver = AseFrequencyDriver(calc, params, directory=tmp_path)
    driver.run(atoms)
    leaf = calc.mixer.calcs[0] if change == "mixer" else calc
    assert len(leaf.evaluated_positions) == 3 * params.get("nfree", 2) + 2


def test_mixer_child_parameters_invalidate_cache(tmp_path):
    from gdpx.providers.ase.backend import EnhancedCalculator
    atoms = Atoms("Cu")
    first, second = HarmonicCalculator(), HarmonicCalculator()
    calc = EnhancedCalculator([first, second], weights=[1, 0.5])
    AseFrequencyDriver(calc, {}, directory=tmp_path).run(atoms)
    first.evaluated_positions.clear()
    second.parameters["changed"] = True
    AseFrequencyDriver(calc, {}, directory=tmp_path).run(atoms)
    assert len(first.evaluated_positions) == 8


def test_legacy_cache_requires_explicit_fresh_run(tmp_path):
    atoms = Atoms("Cu")
    AseFrequencyDriver(HarmonicCalculator(), {}, directory=tmp_path).run(atoms)
    (tmp_path / "vibrations.restart.json").unlink()
    original = (tmp_path / "vibrations/cache.eq.json").read_bytes()
    with pytest.raises(RuntimeError, match="no valid restart manifest"):
        AseFrequencyDriver(HarmonicCalculator(), {}, directory=tmp_path).run(atoms)
    assert (tmp_path / "vibrations/cache.eq.json").read_bytes() == original
    calc = HarmonicCalculator()
    AseFrequencyDriver(calc, {"resume": False}, directory=tmp_path).run(atoms)
    assert len(calc.evaluated_positions) == 8


def test_input_file_change_invalidates_cache(tmp_path):
    input_file = tmp_path / "model.dat"
    input_file.write_text("first")
    calc = HarmonicCalculator()
    calc.parameters["model"] = str(input_file)
    AseFrequencyDriver(calc, {}, directory=tmp_path).run(Atoms("Cu"))
    calc.evaluated_positions.clear()
    input_file.write_text("second")
    AseFrequencyDriver(calc, {}, directory=tmp_path).run(Atoms("Cu"))
    assert len(calc.evaluated_positions) == 8
