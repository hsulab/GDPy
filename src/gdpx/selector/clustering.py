#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
from typing import Optional

from gdpx.data.array import AtomsNDArray

from .utils import get_aligned_chemical_formula


def group_structures(structures: AtomsNDArray, group_by: Optional[str] = None):
    """Group structures by the specified criteria."""
    if group_by is not None:
        if group_by.isdigit():  # For backward compatibility,
            axis = int(group_by)
            marker_groups = group_structures_by_axis(structures, axis=axis)
        elif group_by.startswith("axis"):
            axis = int(group_by.strip().split()[1])
            marker_groups = group_structures_by_axis(structures, axis=axis)
        elif group_by.startswith("chemical_formula"):
            params = group_by.strip().split()
            symbol_list, padding_width = None, 4
            num_params = len(params)
            if num_params == 2:
                if params[1].isdigit():
                    padding_width = int(params[1])
                else:
                    symbol_list = [params[1]]
            elif num_params >= 3:
                symbol_list = params[1:-1]
                padding_width = int(params[-1])
            else:
                ...
            marker_groups = group_structures_by_chemical_formula(structures, symbol_list, padding_width)
        else:
            raise Exception(f"Unsupported group_by {group_by}.")
    else:
        marker_groups = dict(all=structures.markers)

    return marker_groups


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


def group_structures_by_chemical_formula(
    structures: AtomsNDArray, symbol_list: Optional[list[str]] = None, padding_width: int = 4
):
    """"""
    # Set the function to get the chemical formula
    if symbol_list is None:
        func = lambda x: structures[tuple(x.tolist())].get_chemical_formula()
    else:
        func = lambda x: get_aligned_chemical_formula(structures[tuple(x.tolist())], symbol_list, padding_width)

    # Group by the chemical formula
    marker_groups = {}
    for k, v in itertools.groupby(structures.markers, key=func):
        if k in marker_groups:
            marker_groups[k].extend(list(v))
        else:
            marker_groups[k] = list(v)

    return marker_groups


if __name__ == "__main__":
    ...
