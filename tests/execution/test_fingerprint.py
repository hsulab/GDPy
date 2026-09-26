import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixCartesian
from ase.io.jsonio import decode, encode

from gdpx.execution.fingerprint import (
    structure_digest, payload_digest, read_structure_inputs, write_structure_inputs,
)
from gdpx.execution.workers.utils import copy_minimal_frames


def structure():
    atoms = Atoms("CuH", positions=[[0.12345678901234567, 0, 0], [1, 2, 3]],
                  cell=[4, 5, 6], pbc=[True, False, True])
    return atoms


@pytest.mark.parametrize("field", [
    "positions", "numbers", "cell", "pbc", "tags", "masses", "momenta",
    "initial_charges", "initial_magmoms", "constraints", "custom",
])
def test_calculation_inputs_change_fingerprint(field):
    original = structure()
    changed = original.copy()
    if field == "positions":
        changed.positions[0, 0] = np.nextafter(changed.positions[0, 0], np.inf)
    elif field == "numbers":
        changed.numbers[0] = 28
    elif field == "cell":
        changed.cell[0, 0] += 0.1
    elif field == "pbc":
        changed.pbc[1] = True
    elif field == "constraints":
        changed.set_constraint(FixAtoms(indices=[0]))
    else:
        shape = (2, 3) if field == "momenta" else (2,)
        changed.set_array(field, np.ones(shape, dtype=int if field == "tags" else float))
    assert structure_digest([original]) != structure_digest([changed])


def test_order_matters_but_bookkeeping_and_results_do_not():
    first = structure()
    second = first.copy()
    second.positions[1, 0] += 0.1
    expected = structure_digest([first, second])
    assert expected != structure_digest([second, first])
    assert structure_digest([first]) != structure_digest([first[::-1]])
    first.info.update(wdir="cand5", step=90, note=object())
    first.calc = SinglePointCalculator(first, energy=-4)
    assert structure_digest([first, second]) == expected


def test_defaults_endianness_memory_order_and_signed_zero():
    first = structure()
    expected = structure_digest([first])
    second = first.copy()
    second.set_tags([0, 0])
    second.set_masses(first.get_masses())
    second.set_momenta(np.zeros((2, 3)))
    second.set_initial_charges([0.0, 0.0])
    second.set_initial_magnetic_moments([0.0, 0.0])
    second.arrays["positions"] = np.asfortranarray(second.positions.astype(">f8"))
    second.arrays["numbers"] = second.numbers.astype(">i4")
    second.positions[0, 1] = -0.0
    assert structure_digest([second]) == expected
    assert payload_digest({"b": np.int64(3), "a": np.array([1, 2])}) == payload_digest(
        {"a": [1, 2], "b": 3}
    )


def test_lossless_snapshot_and_minimal_copy(tmp_path):
    original = structure()
    original.set_constraint([FixAtoms(indices=[0]), FixCartesian(1, mask=[True, False, False])])
    original.set_momenta(np.arange(6).reshape(2, 3), apply_constraint=False)
    original.set_initial_charges([0.1, -0.1])
    original.set_initial_magnetic_moments([[1, 0, 0], [0, 1, 0]])
    original.set_masses([65, 2])
    original.set_tags([5, 6])
    original.set_array("custom", np.array([[2.3], [4.5]]))
    original.info["wdir"] = "source"
    frames, _ = copy_minimal_frames([original])
    assert frames[0].info == {}
    assert structure_digest(frames) == structure_digest([original])
    path = tmp_path / "input.atoms.json"
    digest = write_structure_inputs(path, frames)
    restored = read_structure_inputs(path, digest)
    assert np.array_equal(original.positions, restored[0].positions)
    assert structure_digest(restored) == digest
    assert len(restored[0].constraints) == 2
    restored[0].positions[0, 0] = 7
    assert original.positions[0, 0] != 7


def test_snapshot_corruption_and_unknown_version_rejected(tmp_path):
    path = tmp_path / "input.atoms.json"
    write_structure_inputs(path, [structure()])
    saved = decode(path.read_text())
    saved["frames"][0].positions[0, 0] += 1e-12
    path.write_text(encode(saved))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        read_structure_inputs(path)
    saved["version"] = 999
    path.write_text(encode(saved))
    with pytest.raises(ValueError, match="version"):
        read_structure_inputs(path)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_rejected(value):
    atoms = structure()
    atoms.positions[0, 0] = value
    with pytest.raises(ValueError, match="Non-finite"):
        structure_digest([atoms])


def test_snapshots_exclude_unserializable_bookkeeping(tmp_path):
    atoms = structure()
    atoms.info["temporary"] = object()
    atoms.calc = SinglePointCalculator(atoms, energy=-1)
    path = tmp_path / "input.atoms.json"
    write_structure_inputs(path, [atoms])
    restored = read_structure_inputs(path)
    assert restored[0].info == {}
    assert restored[0].calc is None
    assert structure_digest(restored) == structure_digest([atoms])
