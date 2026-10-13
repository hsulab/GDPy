import h5py
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from gdpx.data.array import AtomsNDArray
from gdpx.execution.fingerprint import structure_digest
from gdpx.execution.workers.utils import copy_minimal_frames


def _frame(natoms):
    atoms = Atoms("H" * natoms, positions=np.arange(natoms * 3).reshape(-1, 3) / 7,
                  cell=[5, 6, 0], pbc=[True, True, False])
    atoms.new_array("type", np.arange(natoms, dtype=np.int32) + 1)
    atoms.new_array("custom", np.arange(natoms * 2).reshape(-1, 2) / 11)
    atoms.set_masses(np.arange(natoms) + 2.5)
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.set_momenta(np.arange(natoms * 3).reshape(-1, 3) / 13,
                      apply_constraint=False)
    atoms.info.update(step=123, ignored="not retained")
    atoms.calc = SinglePointCalculator(atoms, energy=1.25,
                                       forces=np.ones((natoms, 3)))
    return atoms


def test_cache_preserves_calculation_fingerprint_across_resume(tmp_path):
    frames = [_frame(2), _frame(3)]
    array = AtomsNDArray([[[frames[0], frames[1]]]],
                         dims=("temperature", "structure", "frame"))
    path = tmp_path / "dataset.h5"
    array.save_file(path)
    restored = AtomsNDArray.from_file(path)
    current = restored.get_marked_structures()
    before, _ = copy_minimal_frames(frames)
    after, _ = copy_minimal_frames(current)
    assert structure_digest(before) == structure_digest(after)
    assert restored.shape == array.shape
    assert restored.dims == array.dims
    for original, cached in zip(frames, current):
        assert set(cached.arrays) == set(original.arrays)
        assert cached.arrays["type"].dtype == original.arrays["type"].dtype
        np.testing.assert_array_equal(cached.get_momenta(), original.get_momenta())
        assert cached.info["step"] == 123
        assert "ignored" not in cached.info
        assert cached.get_potential_energy() == original.get_potential_energy()
        np.testing.assert_array_equal(cached.get_forces(apply_constraint=False),
                                      original.get_forces(apply_constraint=False))
    # A second cache round trip must also be stable.
    restored.save_file(path)
    assert structure_digest(AtomsNDArray.from_file(path).get_marked_structures()) == structure_digest(frames)


def test_legacy_hdf5_cache_remains_readable(tmp_path):
    frame = _frame(2)
    path = tmp_path / "legacy.h5"
    AtomsNDArray([frame]).save_file(path)
    with h5py.File(path, "a") as handle:
        del handle["images/structure_inputs"]
    restored = AtomsNDArray.from_file(path).get_marked_structures()[0]
    np.testing.assert_array_equal(restored.positions, frame.positions)
    assert restored.get_potential_energy() == frame.get_potential_energy()
    assert "type" not in restored.arrays
