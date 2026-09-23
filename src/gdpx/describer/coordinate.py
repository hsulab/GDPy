#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy.typing

from gdpx.group import evaluate_group_expression

from .describer import BaseDescriber

COMPONENT_MAP = dict(x=0, y=1, z=2)


class CoordinateDescriber(BaseDescriber):
    """This class describes the coordinates of a group of atoms in a structure."""

    name: str = "distance"

    def __init__(self, group: str, component: str, *args, **kwargs):
        """Initialise the CoordinateDescriber.

        Args:
            group: The group expression to evaluate.
            component: The coordinate component to describe, must be one of "x", "y", or "z".

        """
        super().__init__(*args, **kwargs)

        self.group = group

        if component not in ["x", "y", "z"]:
            raise RuntimeError("Coordinate component must be x or y or z.")
        self.component = component

        return

    def run(self, structures) -> numpy.typing.NDArray:
        """This method describes the coordinates of a group of atoms in a structure."""
        component = COMPONENT_MAP[self.component]

        coordinates = []
        for atoms in structures:
            group_indices = evaluate_group_expression(atoms, self.group)
            group_coordinates = atoms.positions[group_indices, component].flatten()
            coordinates.append(group_coordinates)

        coordinates = np.array(coordinates)

        return coordinates

if __name__ == "__main__":
    ...
