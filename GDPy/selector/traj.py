#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from pathlib import Path
import copy

import numpy as np

from ase import Atoms

from GDPy.selector.selector import AbstractSelector


class BoltzmannMinimaSelection(AbstractSelector):

    default_parameters = dict(
        fmax = 0.05, # eV
        boltzmann = 3, # kT, eV
        number = [4, 0.2]
    )

    def __init__(self, directory=Path.cwd(), *args, **kwargs):
        """"""
        self.directory = directory

        for k in self.default_parameters:
            if k in kwargs.keys():
                self.default_parameters[k] = kwargs[k]

        self.fmax = self.default_parameters["fmax"] # eV
        self.boltzmann = self.default_parameters["boltzmann"]

        return
    
    def select(self, frames, index_map=None, ret_indices: bool=False, *args, **kwargs) -> List[Atoms]:
        """"""
        super().select(*args,**kwargs)
        # - find minima
        #print("nframes: ", len(frames))
        converged_indices = []
        for i, atoms in enumerate(frames): 
            # check energy if too large then skip
            confid = atoms.info["confid"]
            # cur_energy = atoms.get_potential_energy()
            # if np.fabs(cur_energy - min_energy) > self.ENERGY_DIFFERENCE:
            #     print("Skip high-energy structure...")
            #     continue
            # LAMMPS or LASP has no forces for fixed atoms
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            #print(maxforce, self.fmax)
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
                    converged_energies, converged_indices, num_fixed
                )
            else:
                selected_indices = converged_indices[:num_fixed]
        else:
            selected_indices = []

        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]

        if not ret_indices:
            selected_frames = [frames[i] for i in selected_indices]
            return selected_frames
        else:
            return selected_indices

    def _boltzmann_select(self, props, input_indices, num_minima):
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
            if self.boltzmann > 0.0:
                p *= np.exp(-(H-min_prop)/self.boltzmann)
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
            rv = np.random.uniform()
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

    def _parse_selection_number(self, nframes):
        """ nframes - number of frames
            sometimes maybe zero
        """
        number_info = self.default_parameters["number"]
        if isinstance(number_info, int):
            num_fixed, num_percent = number_info, 0.2
        elif isinstance(number_info, float):
            num_fixed, num_percent = 16, number_info
        else:
            assert len(number_info) == 2, "Cant parse number for selection..."
            num_fixed, num_percent = number_info
        
        if num_fixed is not None:
            if num_fixed > nframes:
                num_fixed = int(nframes*num_percent)
        else:
            num_fixed = int(nframes*num_percent)

        return num_fixed