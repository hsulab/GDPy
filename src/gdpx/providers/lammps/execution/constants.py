import dataclasses

from ase import Atoms


@dataclasses.dataclass(frozen=True)
class AseLammpsSettings:
    """File names for LAMMPS input/output."""

    inputstructure_filename: str = "stru.data"
    trajectory_filename: str = "traj.dump"
    input_fname: str = "in.lammps"
    log_filename: str = "lmp.out"
    deviation_filename: str = "model_devi.out"
    prism_filename: str = "ase-prism.bindat"


ASELMPCONFIG = AseLammpsSettings()


def parse_type_list(atoms: Atoms) -> list[str]:
    """Parse the type list based on input atoms."""
    type_list = list(set(atoms.get_chemical_symbols()))
    type_list.sort()
    return type_list
