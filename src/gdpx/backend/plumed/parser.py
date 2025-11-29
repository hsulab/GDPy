import io
from typing import Optional

import numpy as np
from ase import Atoms


def parse_colvar_data(file_io: io.TextIOBase) -> tuple[np.ndarray, list[str]]:
    """Read COLVAR with one simulation data."""
    # Read column names
    col_names = file_io.readline().split()[2:]
    file_io.seek(0)

    colvars = np.loadtxt(file_io)

    return colvars, col_names


def add_colvar_to_atoms_info(
    file_io: io.TextIOBase, frames: list[Atoms], ignored_columns: Optional[list[str]] = None
) -> None:
    """Add colvar data to atoms.info."""
    colvars, col_names = parse_colvar_data(file_io)

    ignored_columns = ignored_columns or []

    num = min(colvars.shape[0], len(frames))
    for k, v in zip(col_names, colvars.transpose()):
        if k in ignored_columns:
            continue
        for i in range(num):
            frames[i].info[k] = v[i]

    return
