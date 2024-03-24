#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from ..data.array import AtomsNDArray
from ..data.extatoms import ScfErrAtoms
from .selector import AbstractSelector


class ScfSelector(AbstractSelector):

    name = "scf"

    default_parameters = dict(scf_converged=True)

    def _mark_structures(self, data: AtomsNDArray, *args, **kwargs) -> None:
        """"""
        markers, structures = data.markers, data.get_marked_structures()
        if self.parameters["scf_converged"]:
            selected_indices = [
                i for i, a in enumerate(structures) if not isinstance(a, ScfErrAtoms)
            ]
        else:
            selected_indices = [
                i for i, a in enumerate(structures) if not isinstance(a, ScfErrAtoms)
            ]
        selected_markers = [markers[i] for i in selected_indices]

        data.markers = np.array(selected_markers)

        return


if __name__ == "__main__":
    ...
