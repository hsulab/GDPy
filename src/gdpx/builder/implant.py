#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from ase import Atoms
from ase.io import read

from gdpx.group import evaluate_group_expression

from .builder import StructureModifier


class ImplantModifier(StructureModifier):
    """Implant a cluster on the substrate."""

    name: str = "implant"

    def __init__(self, slab: str, cluster_group: str, substrates=None, *args, **kwargs):
        """Initialise the modifier.

        Args:
            slab: The slab to implant the cluster.
            cluster_group: The group expression for getting the cluster.

        """
        super().__init__(substrates=substrates, *args, **kwargs)

        input_slabs = read(slab, ":")
        num_slabs = len(input_slabs)
        if num_slabs != 1:
            raise Exception(f"ImplantModifier only supports one slab, but got {num_slabs}.")
        self.slab = input_slabs[0]

        self.cluster_group = cluster_group

        return

    def run(self, substrates=None, size: int = 1, *args, **kwargs) -> list[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            group_indices = evaluate_group_expression(substrate, self.cluster_group)
            cluster = substrate[group_indices]
            frame = copy.deepcopy(self.slab)
            frame += cluster  # type: ignore
            frames.append(frame)

        return frames


if __name__ == "__main__":
    ...
