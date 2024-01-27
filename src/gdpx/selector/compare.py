#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from ase.io import read, write

from . import registers
from .selector import AbstractSelector


class CompareSelector(AbstractSelector):

    name: str = "compare"

    default_parameters: dict = dict(
        comparator_name = None,
        comparator_params = {}
    )

    def __init__(self, directory="./", axis=None, *args, **kwargs) -> None:
        """"""
        super().__init__(directory, axis, *args, **kwargs)

        self.comparator = registers.create(
            "comparator", self.comparator_name, convert_name=True, 
            **self.comparator_params
        )

        return
    
    def _mark_structures(self, data, *args, **kwargs) -> None:
        """"""
        super()._mark_structures(data, *args, **kwargs)
        
        # -
        structures = data.get_marked_structures()
        nstructures = len(structures)

        # - start from the first structure and compare structures by a given comparator
        if not hasattr(self.comparator, "prepare_data"):
            selected_indices, scores = [0], []
            for i, a1 in enumerate(structures[1:]):
                # NOTE: assume structures are sorted by energy
                #       close structures may have a high possibility to be similar
                #       so we compare reversely
                for j in selected_indices[::-1]:
                    self._print(f"compare: {i+1} and {j}")
                    a2 = structures[j]
                    if self.comparator(a1, a2):
                        break
                else:
                    selected_indices.append(i+1)
                    self._print(f"--->>> current indices: {selected_indices}")
        else:
            fingerprints = self.comparator.prepare_data(structures)

            selected_indices, unique_groups, scores = [], {}, []
            for i, fp in enumerate(fingerprints):
                for j in selected_indices[::-1]:
                    if self.comparator(fp, fingerprints[j]):
                        unique_groups[j].append(i)
                        break
                else:
                    selected_indices.append(i)
                    unique_groups[i] = [i]
            
            unique_wdir = self.directory / "unique"
            unique_wdir.mkdir(parents=True, exist_ok=True)
            
            with open(unique_wdir/"unique.dat", "w") as fopen:
                for k, v in unique_groups.items():
                    fopen.write(f"{k}_{len(v)}: {v}\n")
                    write(unique_wdir/f"g_{k}.xyz", [structures[i] for i in v])

        curr_markers = data.markers
        # NOTE: convert to np.array as there may have 2D markers
        selected_markers = np.array([curr_markers[i] for i in selected_indices])
        data.markers = selected_markers

        return


if __name__ == "__main__":
    ...