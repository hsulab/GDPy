#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, List

import numpy as np

from pathlib import Path

from ase import Atoms
from ase.io import read, write

from GDPy.selector.abstract import AbstractSelector


class DeviationSelector(AbstractSelector):

    # TODO: should be arbitrary property deviation
    # not only energy and force

    name = "deviation"
    selection_criteria = "deviation"

    deviation_criteria = dict(
        # energy tolerance would be natoms*atomic_energy
        atomic_energy = [None, None],
        # maximum fractional force deviation in the system
        force = [None, None] 
    )

    def __init__(
        self,
        properties: dict,
        criteria: dict,
        directory = Path.cwd(),
        pot_worker = None # to access committee
    ):
        self.deviation_criteria = criteria
        self.__parse_criteria()

        self.directory = directory

        self.pot_worker = pot_worker

        # - parse properties
        #self.__register_potential(potential)

        # TODO: select on properties not only on fixed name (energy, forces)
        # if not set properly, will try to call calculator
        self.energy_tag = properties["atomic_energy"]
        self.force_tag = properties["force"]

        return
    
    def __parse_criteria(self):
        """"""
        use_ae, use_force = True, True

        emin, emax = self.deviation_criteria["atomic_energy"]
        fmin, fmax = self.deviation_criteria["force"]

        if emin is None and emax is None:
            use_ae = False
        if emin is None:
            self.deviation_criteria["atomic_energy"][0] = -np.inf
        if emax is None:
            self.deviation_criteria["atomic_energy"][1] = np.inf

        if fmin is None and fmax is None:
            use_force = False
        if fmin is None:
            self.deviation_criteria["force"][0] = -np.inf
        if fmax is None:
            self.deviation_criteria["force"][1] = np.inf

        emin, emax = self.deviation_criteria["atomic_energy"]
        fmin, fmax = self.deviation_criteria["force"]

        if (emin > emax):
            raise RuntimeError("emax should be larger than emin...")
        if (fmin > fmax):
            raise RuntimeError("fmax should be larger than fmin...")
        
        if not (use_ae or use_force):
            raise RuntimeError("neither energy nor force criteria is set...")
        
        self.use_ae = use_ae
        self.use_force = use_force

        return
    
    def select(self, frames, index_map=None, ret_indices: bool=False) -> List[Atoms]:
        """"""
        try:
            self.pfunc("Read uncertainty from frames' info...")
            energy_deviations = [a.info[self.energy_tag] for a in frames] # TODO: max_devi_e is the max atomic en devi
            force_deviations = [a.info[self.force_tag] for a in frames] # TODO: may not exist
            selected_indices = self._select_indices(energy_deviations, force_deviations)
        except KeyError:
            if self.pot_worker:
                committee = getattr(self.pot_worker.potter, "committee", None)
                if committee:
                    self.pfunc("Estimate uncertainty by committee...")
                    frames = self.pot_worker.potter.estimate_uncertainty(frames)
                    write(self.directory/"frames_devi.xyz", frames)
                    energy_deviations = [a.info[self.energy_tag] for a in frames] # TODO: max_devi_e is the max atomic en devi
                    force_deviations = [a.info[self.force_tag] for a in frames] # TODO: may not exist
                    selected_indices = self._select_indices(energy_deviations, force_deviations)
                else:
                    self.pfunc("Could not estimate uncertainty...")
                    selected_indices = list(range(len(frames)))
            else:
                self.pfunc("Could not find deviations of target properties...")
                selected_indices = list(range(len(frames)))

        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]
        
        if not ret_indices:
            selected_frames = [frames[i] for i in selected_indices]
            return selected_frames
        else:
            return selected_indices

    def _select_indices(self, energy_deviations, force_deviations = None) -> List[int]:
        """
        """
        if force_deviations is not None:
            assert len(energy_deviations) == len(force_deviations), "shapes of energy and force deviations are inconsistent"
        else:
            force_deviations = np.empty(len(energy_deviations))
            force_deviations[:] = np.NaN

        emin, emax = self.deviation_criteria["atomic_energy"]
        fmin, fmax = self.deviation_criteria["force"]
        
        # NOTE: deterministic selection
        selected = []
        for idx, (en_devi, force_devi) in enumerate(zip(energy_deviations, force_deviations)):
            if self.use_ae:
                if emin < en_devi < emax:
                    selected.append(idx)
                    continue
            if self.use_force:
                if fmin < force_devi < fmax:
                    selected.append(idx)
                    continue

        return selected
    

if __name__ == "__main__":
    pass