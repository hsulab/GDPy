#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np 
import numpy.ma as ma

from ase import Atoms 
from ase.io import read, write 

from ase.neighborlist import NeighborList 
from ase.neighborlist import natural_cutoffs 

from ase.calculators.calculator import (
    Calculator, all_changes, PropertyNotImplementedError
)
import deepmd.DeepPot as DeepPot


class DP(Calculator):
    name = "DP"
    implemented_properties = ["energy", "forces", "virial", "stress"]

    def __init__(self, model, label="DP", type_dict=None, **kwargs):
        Calculator.__init__(self, label=label, **kwargs)
        if isinstance(model, str):
            model = [model] # compatiblity
        self.dp_models = [] 
        for m in model:
            self.dp_models.append(DeepPot(m))
        if type_dict:
            self.type_dict=type_dict
        else:
            self.type_dict = dict(zip(self.dp.get_type_map(), range(self.dp.get_ntypes())))
    
    def prepare_input(self, atoms=None):
        coord = atoms.get_positions().reshape([1, -1])
        if sum(atoms.get_pbc())>0:
           self.pbc = True
           cell = atoms.get_cell().reshape([1, -1])
        else:
           self.pbc = False
           cell = None
        symbols = atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]

        self.dp_input = {'coords': coord, 'cells': cell, 'atom_types': atype}

        return 

    def icalculate(self, dp_model, properties=["energy", "forces", "virial"], system_changes=all_changes):
        """"""
        results = {}
        e, f, v = dp_model.eval(**self.dp_input)
        results['energy'] = e[0][0]
        results['forces'] = f[0]
        results['virial'] = v[0].reshape(3,3)

        # convert virial into stress for lattice relaxation
        if "stress" in properties:
            if self.pbc:
                # the usual convention (tensile stress is positive)
                # stress = -virial / volume
                stress = -0.5*(v[0].copy()+v[0].copy().T) / atoms.get_volume()
                # Voigt notation 
                results['stress'] = stress.flat[[0,4,8,5,2,1]] 
            else:
                raise PropertyNotImplementedError
        
        return results
    
    def calculate(self, atoms=None, properties=["energy", "forces", "virial"], system_changes=all_changes):
        """"""
        self.prepare_input(atoms)
        all_results = []
        for dp_model in self.dp_models:
            cur_results = self.icalculate(dp_model) # return current results
            all_results.append(cur_results)
        
        if len(self.dp_models) == 1:
            self.results = all_results[0]
        else:
            # average results
            results = {}
            energy_array = [r['energy'] for r in all_results]
            results['energy'] = np.mean(energy_array)
            forces_array = np.array([r['forces'] for r in all_results])
            results['forces'] = np.mean(forces_array, axis=0)
            if 'stress' in properties:
                if self.pbc:
                    stress_array = np.array([r['stress'] for r in all_results])
                    results['stress'] = np.mean(stress_array, axis=0)
                else:
                    raise PropertyNotImplementedError 
            # estimate standard variance
            results['energy_stdvar'] = np.sqrt(np.var(energy_array))
            results['forces_stdvar'] = np.sqrt(np.var(forces_array, axis=0))
            self.results = results

        return 


def use_neighbors():

    #covlent_cutoffs = natural_cutoffs(atoms) 
    #nl = NeighborList(covlent_cutoffs)
    #nl.update(atoms)
    #indices, offsets = nl.get_neighbors(0)

    pass 


def cut_local_environment(atoms, centre_index, cutoff_radius=8.0, box_length=20.): 
    """""" 
    # get necessities 
    cell = atoms.cell
    frac_poses = atoms.get_scaled_positions()
    #print(frac_poses)
    #np.savetxt('n_large.txt',frac_poses)
    numbers = list(atoms.get_atomic_numbers())

    centre_frac_pos = frac_poses[centre_index]
    print('centre position: ', centre_frac_pos)
    centre_cart_pos = np.dot(frac_poses[centre_index], cell)
    # form a 3x3 box 
    offsets = np.array(
        [
            [0, 0, 0], [1, 0, 0], [-1, 0, 0], 
            [0, 1, 0], [1, 1, 0], [-1, 1, 0], 
            [0, -1, 0], [1, -1, 0], [-1, -1, 0], 
            [0, 0, 1], [1, 0, 1], [-1, 0, 1], 
            [0, 1, 1], [1, 1, 1], [-1, 1, 1], 
            [0, -1, 1], [1, -1, 1], [-1, -1, 1], 
            [0, 0, -1], [1, 0, -1], [-1, 0, -1], 
            [0, 1, -1], [1, 1, -1], [-1, 1, -1], 
            [0, -1, -1], [1, -1, -1], [-1, -1, -1]
        ]
    )
    
    extended_frac_poses = np.concatenate(
        (offsets.reshape(27,1,3)+frac_poses.reshape(1,-1,3)), axis=0
    )
    #print(extended_frac_poses.shape)
    extended_numbers = np.array(numbers*27)
    #print(extended_numbers)
    
    # calculate interatomic distances 
    frac_vectors = extended_frac_poses - centre_frac_pos 
    distances = np.linalg.norm(np.dot(frac_vectors, cell), axis=1)
    
    masked_distances = ma.masked_greater_equal(distances, cutoff_radius)
    #print(masked_distances.mask)
    
    # form a cluster 
    dis_mask = masked_distances.mask
    valid_positions = np.dot(
        np.array([extended_frac_poses[i] for i, b in enumerate(dis_mask) if not b]), cell
    )
    valid_positions = valid_positions - centre_cart_pos + np.ones(3)*box_length/2. # centre the cluster 
    #print(valid_positions.shape)
    valid_numbers = ma.array(extended_numbers, mask=masked_distances.mask).compressed() 
    #print(valid_numbers.shape)
    if box_length < 2*(cutoff_radius+5.):
        warnings.warn('The box may be too small for the cluster.', UserWarning)
    cluster_atoms = Atoms(
        numbers=valid_numbers, positions=valid_positions, cell=np.eye(3)*box_length
    )

    return cluster_atoms

def find_maxfstd(forces_stdvar, ifrac=False):
    """"""
    if ifrac:
        max_idx = np.argmax(forces_stdvar)
        irow = max_idx // 3
        icol = max_idx % 3
        atom_idx = irow
        max_fstdvar = forces_stdvar[irow,icol]
    else:
        tf_stdvar = np.linalg.norm(np.power(forces_stdvar,2), axis=1)
        atom_idx = np.argmax(tf_stdvar)
        max_fstdvar = tf_stdvar[atom_idx]

    return atom_idx, max_fstdvar

if __name__ == '__main__': 
    # ===== parameters =====
    # read structures 
    frames = read('test.xyz', ':')

    # set calculator
    type_map = {'O': 0, 'Pt': 1}
    model = [
        '/users/40247882/projects/oxides/gdp-main/it-0003/ensemble/model-0/graph.pb', 
        '/users/40247882/projects/oxides/gdp-main/it-0003/ensemble/model-1/graph.pb', 
        '/users/40247882/projects/oxides/gdp-main/it-0003/ensemble/model-2/graph.pb', 
        '/users/40247882/projects/oxides/gdp-main/it-0003/ensemble/model-3/graph.pb'
    ]

    calc = DP(model=model, type_dict=type_map)

    # deviate and cleave 
    min_stdv, max_stdv = 0.00, 0.25
    cutoff_radius, box_length = 7.0, 20.

    # ===== end parameters =====

    # main loop
    for frame_idx, atoms in enumerate(frames):
        print('structure %4d' %frame_idx)
        calc.reset()
        atoms.calc = calc
        dummy = atoms.get_forces() # carry out one calculation
        # energy_stdvar = atoms.calc.results.get('energy_stdvar', None) # total energy stdvar
        forces_stdvar = atoms.calc.results.get('forces_stdvar', None) # shape (natoms,3)
        atom_idx, max_fstdvar = find_maxfstd(forces_stdvar, ifrac=True) # use frac or summation of xyz
        if min_stdv < max_fstdvar <  max_stdv:
            print(atom_idx, max_fstdvar, 'cut')
            cluster_atoms = cut_local_environment(atoms, atom_idx, cutoff_radius=cutoff_radius, box_length=box_length) 
            write('cut_stru-%s-%s.xyz' %(str(frame_idx).zfill(4), str(atom_idx).zfill(4)), cluster_atoms)
        else:
            print(atom_idx, max_fstdvar, 'skip')
