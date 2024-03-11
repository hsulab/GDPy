#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

from typing import Optional

import numpy as np

from ..data.array import AtomsNDArray
from .selector import AbstractSelector


class RandomSelector(AbstractSelector):

    name = "random"

    default_parameters = dict(number=[4, 0.2])

    """"""

    def __init__(
        self, directory="./", axis: Optional[int] = None, *args, **kwargs
    ) -> None:
        """"""
        super().__init__(directory, axis, *args, **kwargs)

        return

    def _mark_structures(self, data: AtomsNDArray, *args, **kwargs) -> None:
        """"""
        marker_groups = self.group_structures_by_axis(data, self.axis)

        selected_markers = []
        for grp_name, markers in marker_groups.items():
            num_markers = len(markers)
            num_selected = self._parse_selection_number(num_markers)
            if num_selected > 0:
                curr_selected_markers = self.rng.choice(
                    markers, size=num_selected, replace=False
                )
                selected_markers.extend(curr_selected_markers)
            else:
                ...

        data.markers = np.array(selected_markers)

        return

    @staticmethod
    def group_structures_by_axis(data: AtomsNDArray, axis: Optional[int] = None):
        # - group markers
        if axis is not None:
            ndim = len(data.shape)
            if axis < -ndim or axis > ndim:
                raise IndexError(f"axis {axis} is out of dimension {ndim}.")
            if axis < 0:
                axis = ndim + axis

            marker_groups = {}
            for k, v in itertools.groupby(data.markers, key=lambda x: x[axis]):
                if k in marker_groups:
                    marker_groups[k].extend(list(v))
                else:
                    marker_groups[k] = list(v)
        else:
            marker_groups = dict(all=data.markers)

        return marker_groups


if __name__ == "__main__":
    ...
