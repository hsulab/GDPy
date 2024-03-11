#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from typing import Union, List

import numpy as np

from ase import Atoms

from ..data.array import AtomsNDArray
from .selector import AbstractSelector


class IntervalSelector(AbstractSelector):

    name = "interval"

    default_parameters = dict(
        period=1,
        include_first=True,
        include_last=False,
    )

    """This is a number-unaware selector.
    """

    def __init__(self, directory="./", *args, **kwargs) -> None:
        """"""
        super().__init__(directory, *args, **kwargs)

        return

    def _mark_structures(self, data: AtomsNDArray, *args, **kargs) -> None:
        """Select structures.

        Add unmasks to input trajectories.

        Args:
            inp_dat: Structures.

        """
        # if axis == -1 or axis == ndim-1:
        #    # NOTE: Last dimension is the trajectory
        #    #       it may have padded dummy atoms
        #    ...
        # else:
        #    ...

        # - group markers
        if self.axis is not None:
            axis = self.axis
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

        self._debug(f"marker_groups: {marker_groups}")

        selected_markers = []
        for curr_grpname, curr_markers in marker_groups.items():
            curr_markers = sorted(np.array(curr_markers).tolist())
            nstructures = len(curr_markers)

            first, last = 0, nstructures - 1
            if self.include_first:
                curr_indices = list(range(0, nstructures, self.period))
                if self.include_last:
                    if last not in curr_indices:
                        curr_indices.append(last)
                else:
                    if last in curr_indices:
                        curr_indices.remove(last)
            else:
                curr_indices = list(range(1, nstructures, self.period))
                if self.include_last:
                    if last not in curr_indices:
                        curr_indices.append(last)
                else:
                    if last in curr_indices:
                        curr_indices.remove(last)
            curr_selected_markers = [curr_markers[i] for i in curr_indices]
            selected_markers.extend(curr_selected_markers)

        data.markers = np.array(selected_markers)

        return


if __name__ == "__main__":
    ...
