#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from pathlib import Path
import copy

import numpy as np

from ase import Atoms

from GDPy.selector.selector import AbstractSelector


""" References
    [1] Bernstein, N.; Csányi, G.; Deringer, V. L. 
        De Novo Exploration and Self-Guided Learning of Potential-Energy Surfaces. 
        npj Comput. Mater. 2019, 5, 99.
    [2] Mahoney, M. W.; Drineas, P. 
        CUR Matrix Decompositions for Improved Data Analysis. 
        Proc. Natl. Acad. Sci. USA 2009, 106, 697–702.
"""


class BoltzmannMinimaSelection(AbstractSelector):

    name = "BoltzMinima"

    default_parameters = dict(
        random_seed = None,
        fmax = 0.05, # eV
        boltzmann = 3, # kT, eV
        ae_cut = None, # eV
        number = [4, 0.2]
    )

    def __init__(self, directory=Path.cwd(), *args, **kwargs):
        """"""
        super().__init__(directory, *args, **kwargs)

        return
    
    def _select_indices(self, frames, *args, **kwargs) -> List[int]:
        """"""
        # - find minima
        converged_indices = []
        for i, atoms in enumerate(frames): 
            # - check if energy is too large
            #   compare to a fixed value or TODO: minimum energy
            cur_energy = atoms.get_potential_energy()
            if self.ae_cut:
                if np.fabs(cur_energy - len(atoms)*self.ae_cut) > 0.:
                    continue
            # LAMMPS or LASP has no forces for fixed atoms
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            if maxforce < self.fmax:
                converged_indices.append(i)
        
        # - sort by energies
        converged_indices = sorted(converged_indices, key=lambda i:frames[i].get_potential_energy())
        #print("converged_indices: ", converged_indices)

        # - boltzmann selection
        num_fixed = self._parse_selection_number(len(converged_indices))

        if num_fixed > 0:
            if self.boltzmann > 0:
                converged_energies = [frames[i].get_potential_energy() for i in converged_indices]
                #print("converged_energies: ", converged_energies)
                selected_indices = self._boltzmann_select(
                    self.boltzmann,
                    converged_energies, converged_indices, num_fixed
                )
            else:
                selected_indices = converged_indices[:num_fixed]
        else:
            selected_indices = []
        
        # - output files
        data = []
        for s in selected_indices:
            atoms = frames[s]
            # - add info
            selection = atoms.info.get("selection","")
            atoms.info["selection"] = selection+f"->{self.name}"
            # - gather info
            confid = atoms.info.get("confid", -1)
            natoms = len(atoms)
            ae = atoms.get_potential_energy() / natoms
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            data.append([s, confid, natoms, ae, maxforce])
        if data:
            np.savetxt(
                self.info_fpath, data, 
                fmt="%8d  %8d  %8d  %12.4f  %12.4f",
                #fmt="{:>8d}  {:>8d}  {:>8d}  {:>12.4f}  {:>12.4f}",
                header="{:>6s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}".format(
                    *"index confid natoms AtomicEnergy MaxForce".split()
                ),
                footer=f"random_seed {self.random_seed}"
            )

        return selected_indices

    def _boltzmann_select(self, boltz, props, input_indices, num_minima):
        """"""
        # compute desired probabilities for flattened histogram
        histo = np.histogram(props, bins=10) # hits, bin_edges
        min_prop = np.min(props)
    
        config_prob = []
        for H in props:
            bin_i = np.searchsorted(histo[1][1:], H) # ret index of the bin
            if histo[0][bin_i] > 0.0:
                p = 1.0/histo[0][bin_i]
            else:
                p = 0.0
            if boltz > 0.0:
                p *= np.exp(-(H-min_prop)/boltz)
            config_prob.append(p)
        
        assert len(config_prob) == len(props)
    
        # - select
        props = copy.deepcopy(props)
        input_indices = copy.deepcopy(input_indices)
        selected_indices = []
        for i in range(num_minima):
            # TODO: rewrite by mask 
            config_prob = np.array(config_prob)
            config_prob /= np.sum(config_prob)
            cumul_prob = np.cumsum(config_prob)
            rv = self.rng.uniform()
            config_i = np.searchsorted(cumul_prob, rv)
            #print(converged_trajectories[config_i][0])
            selected_indices.append(input_indices[config_i])
    
            # remove from config_prob by converting to list
            config_prob = list(config_prob)
            del config_prob[config_i]
    
            # remove from other lists
            del props[config_i]
            del input_indices[config_i]
            
        return selected_indices


if __name__ == "__main__":
    pass