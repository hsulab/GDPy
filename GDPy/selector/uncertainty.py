#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Union, List

import numpy as np

from pathlib import Path

from ase import Atoms
from ase.io import read, write

from GDPy.selector.selector import AbstractSelector


class DeviationSelector(AbstractSelector):

    name = "devi"

    default_parameters = dict(
        #properties = dict(
        #    atomic_energy = "max_devi_e",
        #    force = "max_devi_f"
        #),
        #criteria = dict(
        #    atomic_energy = [0.01, 0.25],
        #    force = [0.05, 0.25]
        #)
        criteria = dict(
            max_devi_e = [0.01, 0.25], # atomic_energy
            max_devi_f = [0.05, 0.50], # force
        )
    )

    def __init__(
        self,
        directory = Path.cwd(),
        pot_worker = None, # to access committee
        *args, **kwargs
    ):
        super().__init__(directory, *args, **kwargs)

        self._parse_criteria()

        self.directory = directory

        self.pot_worker = pot_worker

        return
    
    def _parse_criteria(self):
        """"""
        criteria = dict()
        criteria_ = copy.deepcopy(self.criteria)
        for prop_name, bounds in criteria_.items():
            bounds_ = copy.deepcopy(bounds)
            if bounds_[0] is None:
                bounds_[0] = -np.inf
            if bounds_[1] is None:
                bounds_[1] = np.inf
            assert bounds_[0] < bounds_[1], f"{prop_name} has invalid bounds..."
            criteria[prop_name] = bounds_
        
        self.set(criteria=criteria)

        return
    
    def _select_indices(self, frames, *args, **kwargs) -> List[int]:
        """
        """
        nframes = len(frames)

        deviations = {}
        for prop_name in self.criteria.keys():
            try:
                devi = [a.info[prop_name] for a in frames]
            except KeyError:
                # try to use committee
                if self.pot_worker:
                    committee = getattr(self.pot_worker.potter, "committee", None)
                    if committee:
                        self.pfunc("Estimate uncertainty by committee...")
                        # TODO: check if such committee supports prop_name
                        for i, c in enumerate(self.pot_worker.potter.committee):
                            c.directory = self.directory/f"c{i}"
                        frames = self.pot_worker.potter.estimate_uncertainty(frames)
                        write(self.directory/"frames_devi.xyz", frames)
                        try:
                            devi = [a.info[prop_name] for a in frames]
                        except:
                            self.pfunc(f"Cant evaluate deviation of {prop_name}...")
                            devi = [np.NaN]*len(frames)
                    else:
                        devi = [np.NaN]*len(frames)
                else:
                    devi = [np.NaN]*len(frames)
            finally:
                deviations[prop_name] = devi
        
        selected_indices = self._sift_deviations(deviations, nframes)
        
        # - output
        data = []
        for i, s in enumerate(selected_indices):
            atoms = frames[s]
            # - gather info
            confid = atoms.info.get("confid", -1)
            natoms = len(atoms)
            en = atoms.get_potential_energy()
            ae = en / natoms
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            cur_data = [s, confid, natoms, en, ae, maxforce]
            devis = [atoms.info[prop_name] for prop_name in self.criteria.keys()]
            cur_data.extend(devis)
            data.append(cur_data)
        if data:
            ncols = len(data[0])
            col_names = [s+" " for s in self.criteria.keys()]
            np.savetxt(
                self.info_fpath, data, 
                fmt="%8d  %8d  %8d  "+"%12.4f  "*(ncols-3),
                header=("{:>6s}  {:>8s}  {:>8s}  "+"{:>12s}  "*(ncols-3)).format(
                    *("index confid natoms TotalEnergy AtomicEnergy MaxForce ".split()), *col_names
                ),
                #footer=f"random_seed {self.random_seed}"
            )

        return selected_indices
    
    def _sift_deviations(self, deviations: dict, nframes: int):
        """"""
        # NOTE: deterministic selection
        selected_indices = []
        for idx in range(nframes):
            for prop_name, devi in deviations.items():
                cur_devi = devi[idx]
                pmin, pmax = self.criteria[prop_name]
                if pmin < cur_devi < pmax:
                    selected_indices.append(idx)
                    continue

        return selected_indices
    

if __name__ == "__main__":
    pass