#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from gdpx.data.array import AtomsNDArray

from .clustering import group_structures
from .selector import BaseSelector


class IntervalSelector(BaseSelector):
    """Select structures by interval."""

    name = "interval"

    default_parameters = dict(
        period=1,
        include_first=True,
        include_last=False,
    )

    def _mark_structures(self, data: AtomsNDArray) -> None:
        """Select structures.

        Add unmasks to input trajectories.

        Args:
            inp_dat: Structures.

        """
        marker_groups = group_structures(data, group_by=self.group_by)
        self._debug(f"marker_groups: {marker_groups}")

        selected_markers = []
        for curr_grpname, curr_markers in marker_groups.items():
            curr_markers = sorted(np.array(curr_markers).tolist())
            nstructures = len(curr_markers)

            _, last = 0, nstructures - 1
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

            self._print(
                f"group: {curr_grpname} -> "
                + f"number of structures: {len(curr_selected_markers)}"
            )

        data.markers = np.array(selected_markers)

        return


if __name__ == "__main__":
    ...
