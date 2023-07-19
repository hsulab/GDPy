#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

import numpy as np

from ..core.register import registers
from ..data.array import AtomsNDArray
from .selector import AbstractSelector


@registers.selector.register
class LocateSelector(AbstractSelector):

    name = "locate"

    default_parameters = dict(
        indices = ":" # can be single integer, string or a List of integers
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
            for k, v in itertools.groupby(data.markers, key=lambda x: [x[i] for i in range(data.ndim) if i != self.axis]):
                k = tuple(k)
                if k in marker_groups:
                    marker_groups[k].extend(list(v))
                else:
                    marker_groups[k] = list(v)
            selected_markers = []
            for k, v in marker_groups.items():
                v = sorted(np.array(v).tolist())
                selected_markers.extend([v[i] for i in indices])
            self._print(selected_markers)
        else:
            raise RuntimeError(f"Locator does not support array dimension with {data.ndim}")
        
        data.markers = selected_markers

        return


if __name__ == "__main__":
    ...