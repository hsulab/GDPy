#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
from typing import Optional, Union

import numpy as np

from gdpx.data.array import AtomsNDArray


def group_structures_by_axis(
    structures: AtomsNDArray, axis: Optional[int] = None
):
    """Group structures by axis.

    Args:
        structures: The structures to be grouped.
        axis: The axis to group by. If None, all markers will be grouped together.

    """
    if axis is not None:
        ndim = len(structures.shape)
        if axis < -ndim or axis > ndim:
            raise IndexError(f"axis {axis} is out of dimension {ndim}.")
        if axis < 0:
            axis = ndim + axis

        marker_groups = {}
        for k, v in itertools.groupby(
            structures.markers, key=lambda x: x[axis]
        ):
            if k in marker_groups:
                marker_groups[k].extend(list(v))
            else:
                marker_groups[k] = list(v)
    else:
        marker_groups = dict(all=structures.markers)

    return marker_groups


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
