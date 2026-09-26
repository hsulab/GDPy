import collections
from typing import Union

import numpy as np
import numpy.typing
from ase import Atoms


def stat_str2val(stat: Union[str, float], values: numpy.typing.NDArray) -> float:
    """Get a statistics value based on the input float or string.

    Args:
        stat: Statistics name.
        values: A list of scalar values.

    Return:
        The statistics value.

    """
    if isinstance(stat, str):
        if stat == "min":
            v = np.min(values)
        elif stat == "max":
            v = np.max(values)
        elif stat == "mean" or stat == "avg":  # Compatibilty.
            v = np.mean(values)
        elif stat == "std":
            v = np.std(values)
        elif stat == "median":
            v = np.median(values)
        elif stat.startswith("percentile"):
            q = int(stat.strip().split("_")[1])  # should be within 0 and 100
            v = np.percentile(values, q)
        else:
            raise RuntimeError(f"Unknown statistics {stat}.")
    else:
        if stat == -np.inf:
            v = np.min(values)
        elif stat == np.inf:
            v = np.max(values)
        else:  # assume it is a regular number or a numpy scalar
            v = stat

    return float(v)


def get_aligned_chemical_formula(atoms: Atoms, symbol_list: list[str], padding_width: int = 4):
    """"""
    chemical_symbols = atoms.get_chemical_symbols()
    counter = collections.Counter(chemical_symbols)

    chemical_formula = ""
    for s in symbol_list:
        n = counter.get(s, 0)
        chemical_formula += f"{s}{n:>0{padding_width}d}"

    return chemical_formula
