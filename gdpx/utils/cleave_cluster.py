#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np 
import numpy.ma as ma

from ase import Atoms 
from ase.io import read, write 

from ase.neighborlist import NeighborList 
from ase.neighborlist import natural_cutoffs 

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
    numbers = list(atoms.get_atomic_numbers())

    centre_frac_pos = frac_poses[centre_index] 
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

if __name__ == '__main__': 
    # this cuts the local environment of the first atom 
    atoms = read('/users/40247882/repository/gdpx/templates/structures/surf-100_opt.xyz') 
    centre_index = 0
    cluster_atoms = cut_local_environment(atoms, centre_index, cutoff_radius=7.0, box_length=20.) 
    #write('cut.xsd', cluster_atoms)
    write('cut.xyz', cluster_atoms)
