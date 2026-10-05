import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.data.extatoms import ScfErrAtoms


def test_scf_error_atoms_preserves_source_atoms():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
        info={"step": 4},
    )
    atoms.set_momenta([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0)

    converted = ScfErrAtoms.from_atoms(atoms)

    assert isinstance(converted, ScfErrAtoms)
    assert converted.info == {"step": 4, "scf_error": True}
    assert converted.calc is atoms.calc
    assert np.array_equal(converted.positions, atoms.positions)
    assert np.array_equal(converted.get_momenta(), atoms.get_momenta())
    assert np.array_equal(converted.cell, atoms.cell)
    assert np.array_equal(converted.pbc, atoms.pbc)
