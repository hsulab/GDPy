from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from gdpx.data.loaders.dataset import XyzDataloader
from gdpx.providers.nnp.trainer import _materialize_dataset


def test_nnp_materializes_workflow_xyz_dataset(tmp_path):
    system = tmp_path / "seed-Cu-bulk"
    system.mkdir()
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=0.0,
        forces=[[0.0, 0.0, 0.0]],
    )
    write(system / "seed.xyz", atoms)

    frames = _materialize_dataset(XyzDataloader(tmp_path))

    assert len(frames) == 1
    assert frames[0].get_chemical_symbols() == ["Cu"]
    assert frames[0].get_potential_energy() == 0.0
