#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
import operator
from typing import List

import numpy as np


def str2list_int(
    inp: str, convention: str = "lmp", out_convention: str = "ase"
) -> List[int]:
    """Convert a string to a List of int.

    Args:
        inp: A string contains numbers and colons.
        convention: The input convention either `lmp` or `ase`.
                    lmp index starts from 1 and includes the last.

    Examples:
        >>> str2list_int("1:2 4:6", "lmp")
        >>> [0, 1, 3, 4, 5]
        >>> str2list_int("1:2 4:6", "ase")
        >>> [1, 4, 5]

    Returns:
        A List of integers.

    """
    ret = []
    for x in inp.strip().split():
        curr_range = list(map(int, x.split(":")))
        if len(curr_range) == 1:
            start, end = curr_range[0], curr_range[0]
        else:
            start, end = curr_range
        if convention == "lmp":
            ret.extend([i - 1 for i in list(range(start, end + 1))])
        elif convention == "ase":
            ret.extend(list(range(start, end)))
        else:
            ...

    if out_convention == "lmp":
        ret = [r + 1 for r in ret]
    elif out_convention == "ase":
        ...
    else:
        ...

    # Remove duplicates after the final conversion,
    # otherwise, "0:2" in lmp convention will be [1, 2, -1]
    # due to the set sort positive then negative.
    ret = list(set(ret))

    return ret


def integers_to_string(
    indices: list[int],
    inp_convention: str = "ase",
    out_convention: str = "lmp",
) -> str:
    """Convert a list of integers to a string.

    Args:
        indices: A list of integers.
        inp_convention: The input convention either `lmp` or `ase`.
        out_convention: The output convention must be `lmp`.

    Examples:
        >>> integers_to_string([6, 1, 7, 8], "lmp", "lmp")
        >>> "1 6:8"

        >>> integers_to_string([6, 1, 7, 8], "ase", "lmp")
        >>> "2 7:9"

    Returns:
        A string.

    """
    if out_convention != "lmp":
        raise Exception("The output string must be in the ASE convention.")
    indices = sorted(indices)
    if inp_convention == "lmp":
        if 0 in indices:
            raise Exception(
                "The input indices should be greater than 0 in the LAMMPS convention."
            )
    elif inp_convention == "ase":
        indices = [i + 1 for i in indices]

    ret = []
    for _, g in itertools.groupby(enumerate(indices), lambda x: x[0] - x[1]):
        group = map(operator.itemgetter(1), g)
        group = list(map(int, group))
        if group[0] == group[-1]:
            ret.append(str(group[0]))
        else:
            ret.append("{}:{}".format(group[0], group[-1]))
    ret = " ".join(ret)

    return ret


def str2array(inp: str):
    """Convert a string to a np.array using np.arange.

    The endpoint is always included.

    """
    ret = []
    for x in inp.strip().split():
        curr_range = list(map(float, x.split(":")))
        if len(curr_range) == 1:
            start, end, step = curr_range[0], curr_range[0] + 0.01, 1e8
        elif len(curr_range) == 3:
            start, end, step = curr_range
            end += step * 1e-8
        else:
            raise RuntimeError(f"Invalid range `{curr_range}`.")
        ret.extend(np.arange(start, end, step).tolist())

    # Donot sort entries and just keep it as what it is
    # ret = np.array(sorted(ret))
    ret = np.array(ret)

    return ret


if __name__ == "__main__":
    ...
