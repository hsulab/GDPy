#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from gdpx.data.array import AtomsNDArray

from .clustering import group_structures_by_axis
from .selector import BaseSelector


class RandomSelector(BaseSelector):

    name = "random"

    default_parameters = dict(number=[4, 0.2])

    def __init__(self, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

        return

    def _mark_structures(self, data: AtomsNDArray) -> None:
        """"""
        marker_groups = group_structures_by_axis(data, self.group_by)

        selected_markers = []
        for grp_name, markers in marker_groups.items():
            num_markers = len(markers)
            num_selected = self._parse_selection_number(num_markers)
            if num_selected > 0:
                curr_selected_markers = self.rng.choice(
                    markers, size=num_selected, replace=False
                )
                selected_markers.extend(curr_selected_markers)
                self._print(
                    f"group: {grp_name} -> "
                    + f"number of structures: {len(curr_selected_markers)}"
                )
            else:
                ...

        data.markers = np.array(selected_markers)

        return


if __name__ == "__main__":
    ...
