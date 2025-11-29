import io
import itertools

import numpy as np
from ase import Atoms


def parse_model_deviation_data(file_io: io.TextIOBase, units: str) -> tuple[np.ndarray, list[str]]:
    """"""
    # Read column names
    lines = file_io.readline()
    col_names = ("".join([x for x in lines[0] if x != "#"])).strip().split()
    col_names = [x.strip() for x in col_names][1:]
    file_io.seek(0)

    # Convert data
    data = np.loadtxt(file_io, dtype=np.float64)
    num_cols = data.shape[-1]

    data = data.reshape(-1, num_cols)

    return data, col_names


def add_model_deviation_to_atoms_info(file_io: io.TextIOBase, frames: list[Atoms], units: str) -> None:
    """Add model deviation data to atoms.info."""
    assert units == "metal", "Only `metal` units are supported for model deviation data."
    data, col_names = parse_model_deviation_data(file_io, units=units)

    # For some minimisers, dp gives several deviations as
    # multiple force evluations are performed in one step.
    # Thus, we only take the last occurance of the deviation in each step.
    step_indices = []
    steps = data[:, 0].astype(np.int64).tolist()
    for k, v in itertools.groupby(enumerate(steps), key=lambda x: x[1]):
        v = sorted(v, key=lambda x: x[0])
        step_indices.append(v[-1][0])
    data = data.transpose()[1:, step_indices]  # skip `Step` column

    num = min(data.shape[0], len(frames))
    for col_name, values in zip(col_names, data):
        for i in range(num):
            frames[i].info[col_name] = values[i]

    return
