#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
from typing import Optional

from gdpx.data.array import AtomsNDArray


def group_structures_by_axis(structures: AtomsNDArray, axis: Optional[int] = None):
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
        for k, v in itertools.groupby(structures.markers, key=lambda x: x[axis]):
            if k in marker_groups:
                marker_groups[k].extend(list(v))
            else:
                marker_groups[k] = list(v)
    else:
        marker_groups = dict(all=structures.markers)

    return marker_groups


if __name__ == "__main__":
    ...
