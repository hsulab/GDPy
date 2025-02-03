#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union

import numpy as np


def stat_str2val(stat: Union[str, float], values: list[float]) -> float:
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


if __name__ == "__main__":
    ...
