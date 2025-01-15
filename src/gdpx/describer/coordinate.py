#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.group.group import create_a_group

from .describer import AbstractDescriber

COMPONENT_MAP = dict(x=0, y=1, z=2)


class CoordinateDescriber(AbstractDescriber):

    name: str = "distance"

    def __init__(self, group, component: str, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.group = group

        if component not in ["x", "y", "z"]:
            raise RuntimeError("Coordinate component must be x or y or z.")
        self.component = component

        return

    def run(self, structures):
        """"""
        component = COMPONENT_MAP[self.component]

        coordinates = []
        for atoms in structures:
            group_indices = create_a_group(atoms, self.group)
            group_coordinates = atoms.positions[group_indices, component].flatten()
            coordinates.append(group_coordinates)

        return coordinates


if __name__ == "__main__":
    ...
