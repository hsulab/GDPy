#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NoReturn, List

import numpy as np

from ase import Atoms

from GDPy.selector.selector import AbstractSelector


class SortCutSelector(AbstractSelector):

    """Get a given number of structures after their target properties being sorted.
    """

    name = "sortcut"

    default_parameters = dict(
        random_seed = None,
        property = "energy",
        reverse = False, # whether reverse the sorting
        number = [4, 0.2]
    )

    def __init__(self, directory="./", *args, **kwargs) -> NoReturn:
        super().__init__(directory, *args, **kwargs)

        return
    
    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """Returen selected indices."""
        # - get number
        nframes = len(frames)
        num_fixed = self._parse_selection_number(nframes)

        # - get target property
        target_properties = []
        for i, atoms in enumerate(frames):
            if self.property == "energy":
                ene = atoms.get_potential_energy()
                target_properties.append(ene)
            else:
                raise NotImplementedError(f"Not supported property {self.property}.")
                ...
        
        # - sort properties
        numbers = list(range(nframes))
        sorted_numbers = sorted(numbers, key=lambda i: target_properties[i], reverse=self.reverse)

        selected_indices = sorted_numbers[:num_fixed]

        self._save_restart_data(frames, selected_indices, target_properties)

        return selected_indices
    
    def _save_restart_data(self, frames: List[Atoms], selected_indices: List[int], target_properties: List[float]):
        """Save information that helps restart."""
        data = []
        for i, s in enumerate(selected_indices):
            atoms = frames[s]
            # - gather info
            confid = atoms.info.get("confid", -1)
            natoms = len(atoms)
            try:
                ae = atoms.get_potential_energy() / natoms
            except:
                ae = np.NaN
            try:
                maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            except:
                maxforce = np.NaN
            score = target_properties[s]
            data.append([s, confid, natoms, ae, maxforce, score])

        if data:
            np.savetxt(
                self.info_fpath, data, 
                fmt="%8d  %8d  %8d  %12.4f  %12.4f  %12.4f",
                #fmt="{:>8d}  {:>8d}  {:>8d}  {:>12.4f}  {:>12.4f}",
                header="{:>6s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}  {:>12s}".format(
                    *"index confid natoms AtomicEnergy MaxForce  Property".split()
                ),
                footer=f"random_seed {self.random_seed}"
            )
        else:
            np.savetxt(
                self.info_fpath, [[np.NaN]*6],
                header="{:>6s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}  {:>12s}".format(
                    *"index confid natoms AtomicEnergy MaxForce  Property".split()
                ),
                footer=f"random_seed {self.random_seed}"
            )

        return


if __name__ == "__main__":
    ...