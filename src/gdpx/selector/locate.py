#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

import numpy as np

from ..data.array import AtomsNDArray
from .selector import AbstractSelector


def convert_string_to_indices(indstr: str, length: int, convention="py"):
    """"""
    # NOTE: If the input is a valid number, we need convert it to str first.
    indstr = str(indstr)

    def _convert_string(x: str):
        """"""
        try:
            x = int(x)
        except:  # == ""
            x = None

        return x

    # print(f"indstr: {indstr}")
    indices = list(range(length))
    selected_indices = []
    for x in indstr.strip().split():
        curr_range = x.split(":")
        if len(curr_range) == 1:
            x = _convert_string(curr_range[0])
            if x is None:  # == ""
                start, stop, step = None, None, None
            else:
                if x >= 0:
                    start, stop, step = x, x + 1, None
                else:
                    start, stop, step = x, x - 1, -1
        elif len(curr_range) == 2:
            x, y = _convert_string(curr_range[0]), _convert_string(curr_range[1])
            start, stop, step = x, y, None
        elif len(curr_range) == 3:
            x, y, z = (
                _convert_string(curr_range[0]),
                _convert_string(curr_range[1]),
                _convert_string(curr_range[2]),
            )
            start, stop, step = x, y, z
        else:
            raise RuntimeError(f"Fail to parse the index string {x}.")
        # print(f"slice: {slice(start, stop, step)}")
        selected_indices.extend(indices[slice(start, stop, step)])

    return selected_indices


class LocateSelector(AbstractSelector):

    name = "locate"

    default_parameters = dict(
        indices=":"  # can be single integer, string or a List of integers
    )

    def __init__(self, directory="./", axis=None, *args, **kwargs) -> None:
        super().__init__(directory, axis, *args, **kwargs)

        return

    def _mark_structures(self, data: AtomsNDArray, *args, **kwargs) -> None:
        """"""
        super()._mark_structures(data, *args, **kwargs)

        axis = self.axis
        if axis < 0:
            axis = data.ndim + axis
        indices = self.indices

        # This is similar to np.take_along_axis
        if data.ndim > 0:
            marker_groups = {}
            for k, v in itertools.groupby(
                data.markers,
                key=lambda x: [x[i] for i in range(data.ndim) if i != axis],
            ):
                k = tuple(k)
                if k in marker_groups:
                    marker_groups[k].extend(list(v))
                else:
                    marker_groups[k] = list(v)
            selected_markers = []
            for k, v in marker_groups.items():
                v = sorted(np.array(v).tolist())
                selected_markers.extend(
                    [v[i] for i in convert_string_to_indices(self.indices, len(v))]
                )
            # self._print(f"selected_markers: {selected_markers}")
        else:
            raise RuntimeError(
                f"Locator does not support array dimension with {data.ndim}"
            )

        data.markers = selected_markers

        return


if __name__ == "__main__":
    ...
