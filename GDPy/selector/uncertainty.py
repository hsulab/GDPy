#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, List

import numpy as np

from pathlib import Path

from ase import Atoms
from ase.io import read, write

from GDPy.selector.selector import AbstractSelector


class DeviationSelector(AbstractSelector):

    name = "devi"

    default_parameters = dict(
        properties = dict(
            atomic_energy = "max_devi_e",
            force = "max_devi_f"
        ),
        criteria = dict(
            atomic_energy = [0.01, 0.25],
            force = [0.05, 0.25]
        )
    )

    def __init__(
        self,
        directory = Path.cwd(),
        pot_worker = None, # to access committee
        *args, **kwargs
    ):
        super().__init__(directory, *args, **kwargs)

        self.__parse_criteria()

        self.directory = directory

        self.pot_worker = pot_worker

        # TODO: select on properties not only on fixed name (energy, forces)
        # if not set properly, will try to call calculator
        self.energy_tag = self.properties["atomic_energy"]
        self.force_tag = self.properties["force"]

        return
    
    def __parse_criteria(self):
        """"""
        use_ae, use_force = True, True

        emin, emax = self.criteria["atomic_energy"]
        fmin, fmax = self.criteria["force"]

        if emin is None and emax is None:
            use_ae = False
        if emin is None:
            self.criteria["atomic_energy"][0] = -np.inf
        if emax is None:
            self.criteria["atomic_energy"][1] = np.inf

        if fmin is None and fmax is None:
            use_force = False
        if fmin is None:
            self.criteria["force"][0] = -np.inf
        if fmax is None:
            self.criteria["force"][1] = np.inf

        emin, emax = self.criteria["atomic_energy"]
        fmin, fmax = self.criteria["force"]

        if (emin > emax):
            raise RuntimeError("emax should be larger than emin...")
        if (fmin > fmax):
            raise RuntimeError("fmax should be larger than fmin...")
        
        if not (use_ae or use_force):
            raise RuntimeError("neither energy nor force criteria is set...")
        
        self.use_ae = use_ae
        self.use_force = use_force

        return
    
    def _select_indices(self, frames, *args, **kwargs) -> List[int]:
        """
        """
        try:
            self.pfunc("Read uncertainty from frames' info...")
            energy_deviations = [a.info[self.energy_tag] for a in frames] # TODO: max_devi_e is the max atomic en devi
            force_deviations = [a.info[self.force_tag] for a in frames] # TODO: may not exist
            selected_indices = self._sift_deviations(energy_deviations, force_deviations)
        except KeyError:
            if self.pot_worker:
                committee = getattr(self.pot_worker.potter, "committee", None)
                if committee:
                    self.pfunc("Estimate uncertainty by committee...")
                    frames = self.pot_worker.potter.estimate_uncertainty(frames)
                    write(self.directory/"frames_devi.xyz", frames)
                    energy_deviations = [a.info[self.energy_tag] for a in frames] # TODO: max_devi_e is the max atomic en devi
                    force_deviations = [a.info[self.force_tag] for a in frames] # TODO: may not exist
                    selected_indices = self._sift_deviations(energy_deviations, force_deviations)
                else:
                    self.pfunc("Could not estimate uncertainty...")
                    energy_deviations, force_deviations = [np.NaN]*len(frames), [np.NaN]*len(frames)
                    selected_indices = list(range(len(frames)))
            else:
                self.pfunc("Could not find deviations of target properties...")
                energy_deviations, force_deviations = [np.NaN]*len(frames), [np.NaN]*len(frames)
                selected_indices = list(range(len(frames)))
        
        # - output
        data = []
        for i, s in enumerate(selected_indices):
            atoms = frames[s]
            # - add info
            selection = atoms.info.get("selection","")
            atoms.info["selection"] = selection+f"->{self.name}"
            # - gather info
            confid = atoms.info.get("confid", -1)
            natoms = len(atoms)
            ae = atoms.get_potential_energy() / natoms
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            devi_e, devi_f = energy_deviations[i], force_deviations[i]
            data.append([s, confid, natoms, ae, maxforce, devi_e, devi_f])
        if data:
            np.savetxt(
                self.info_fpath, data, 
                fmt="%8d  %8d  %8d  %12.4f  %12.4f  %12.4f  %12.4f",
                header="{:>6s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}".format(
                    *"index confid natoms AtomicEnergy MaxForce DeviAE DeviF".split()
                ),
                #footer=f"random_seed {self.random_seed}"
            )

        return selected_indices
    
    def _sift_deviations(self, energy_deviations, force_deviations):
        """"""
        if force_deviations is not None:
            assert len(energy_deviations) == len(force_deviations), "shapes of energy and force deviations are inconsistent"
        else:
            force_deviations = np.empty(len(energy_deviations))
            force_deviations[:] = np.NaN

        emin, emax = self.criteria["atomic_energy"]
        fmin, fmax = self.criteria["force"]
        
        # NOTE: deterministic selection
        selected_indices = []
        for idx, (en_devi, force_devi) in enumerate(zip(energy_deviations, force_deviations)):
            if self.use_ae:
                if emin < en_devi < emax:
                    selected_indices.append(idx)
                    continue
            if self.use_force:
                if fmin < force_devi < fmax:
                    selected_indices.append(idx)
                    continue

        return selected_indices
    

if __name__ == "__main__":
    pass