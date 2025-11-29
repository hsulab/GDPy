import io
from typing import Optional

import numpy as np
from ase import Atoms


def parse_colvar_data(file_io: io.TextIOBase) -> tuple[dict[str, np.ndarray], int]:
    """Read COLVAR with one simulation data."""
    # Read column names
    col_names = file_io.readline().split()[2:]
    file_io.seek(0)

    colvars = np.loadtxt(file_io)

    num_entries = colvars.shape[0]
    cv_dict = {name: colvars[:, idx] for idx, name in enumerate(col_names)}

    return cv_dict, num_entries


def add_colvar_to_atoms_info(
    file_io: io.TextIOBase, frames: list[Atoms], ignored_columns: Optional[list[str]] = None
) -> None:
    """Add colvar data to atoms.info."""
    cv_dict, num_entries = parse_colvar_data(file_io)

    ignored_columns = ignored_columns or []

    num = min(num_entries, len(frames))
    for k, v in cv_dict.items():
        if k in ignored_columns:
            continue
        for i in range(num):
            frames[i].info[k] = v[i]

    return
