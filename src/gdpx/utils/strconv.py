#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np


def str2list_int(inp: str, convention: str = "lmp") -> List[int]:
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

    # remove duplicates
    ret = sorted(list(set(ret)))

    return ret


def str2array(inp: str):
    """Convert a string to a np.array using np.arange.

    The endpoint is always included.

    """
    ret = []
    for x in inp.strip().split():
        curr_range = list(map(float, x.split(":")))
        if len(curr_range) == 1:
            start, end, step = curr_range[0], curr_range[0] + 0.01, 0.01
        elif len(curr_range) == 3:
            start, end, step = curr_range
        else:
            raise RuntimeError(f"Invalid range `{curr_range}`.")
        ret.extend(np.arange(start, end + 0.01, step).tolist())

    ret = np.array(sorted(ret))

    return ret


if __name__ == "__main__":
    ...
