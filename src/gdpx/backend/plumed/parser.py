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
    file_io: io.TextIOBase,
    frames: list[Atoms],
    dump_period_in_ps: float,
    ignored_columns: Optional[list[str]] = None,
) -> None:
    """Add colvar data to atoms.info."""
    colvars, col_names = parse_colvar_data(file_io)

    num_frames = len(frames)
    print(f"{num_frames=}  {colvars.shape=}")

    num = min(colvars.shape[0], len(frames))

    assert col_names[0] == "time", "The first column must be 'time'."
    time_in_colvar = colvars[:, 0]

    # verify time in traj and colvar match
    time_from_traj = np.arange(num_frames) * dump_period_in_ps
    if not np.allclose(time_in_colvar[:num], time_from_traj[:num]):
        raise Exception("Time in COLVAR does not match time in trajectory. ")

    # add info
    ignored_columns = ignored_columns or []

    for k, v in zip(col_names, colvars.transpose()):
        if k in ignored_columns:
            continue
        for i in range(num):
            frames[i].info[k] = v[i]

    return
