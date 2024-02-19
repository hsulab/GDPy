#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import itertools

import numpy as np

from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs


DimerData = collections.namedtuple("DimerData", ["fi", "i", "j", "pos_i", "pos_j", "shift"])

def compute_body3_descriptor(frames, r_cut: float):
    """"""
    # -- get neighbours
    dimer_list = []

    curr_atomic_index = 0
    for i_frame, atoms in enumerate(frames):
        #print(f"frame: {i_frame}")
        natoms = len(atoms)
        nl = NeighborList(
            cutoffs=[r_cut/2.]*natoms, skin=0.0, sorted=False,
            self_interaction=False, bothways=True
            #self_interaction=False, bothways=False # BUG?
        )
        nl.update(atoms)
        for i in range(natoms):
            nei_indices, nei_offsets = nl.get_neighbors(i)
            for j, offset in zip(nei_indices, nei_offsets):
                pos_i, pos_j = atoms.positions[i, :], atoms.positions[j, :]
                shift = np.dot(offset, atoms.get_cell()) # take negative
                #pos_vec = pos_i - pos_j + shift
                dimer = DimerData(
                    fi = i_frame,
                    i=curr_atomic_index+i, j=curr_atomic_index+j, 
                    pos_i=pos_i, pos_j=pos_j, shift=shift
                    #v=pos_vec, d=np.linalg.norm(pos_vec)
                )
                dimer_list.append(dimer)
        curr_atomic_index += natoms
    
    # - find 2-body
    body2_mapping = [[p.fi, p.i, p.j] for p in dimer_list]
    body2_vectors = np.array([p.pos_i-(p.pos_j+p.shift) for p in dimer_list])
    body2_features = np.linalg.norm(body2_vectors, axis=1)[:, np.newaxis] # (n_dimers, 1)
    body2_gradients = np.concatenate([body2_vectors, -body2_vectors], axis=-1)/body2_features
    body2_gradients = body2_gradients[:, np.newaxis, :] # (num_b2, 1, 6)
    #print("BODY-2: ")
    #print(body2_mapping)
    #print(body2_features)
    #print(body2_gradients)

    # - find 3-body
    trimer_list = []
    for k, v in itertools.groupby(dimer_list, key=lambda x: x.i):
        for pair0, pair1 in itertools.combinations(v, 2):
            pos_i, pos_j = pair0.pos_j + pair0.shift, pair1.pos_j + pair1.shift
            distance = np.linalg.norm(pos_i - pos_j)
            if 1e-8< distance <= r_cut: # j and k may be the same and distance is zero
                pair2 = DimerData(
                    fi=pair0.fi,
                    i=pair0.j, j=pair1.j,
                    pos_i=pos_i, pos_j=pos_j, shift=np.zeros(3)
                )
                trimer_list.append((pair0, pair1, pair2))
    body3_mapping, body3_vectors = [], []
    for pairs in trimer_list:
        body3_mapping.append([pairs[0].fi, pairs[0].i, pairs[0].j, pairs[1].j])
        body3_vectors.append([p.pos_i-(p.pos_j+p.shift) for p in pairs]) 
    if len(body3_vectors) > 0:
        body3_vectors = np.array(body3_vectors) # shape (n_trimers, 3, 3)
        body3_features = np.linalg.norm(body3_vectors, axis=2) # shape (n_trimers, 3)
        body3_gradients = np.concatenate([body3_vectors, -body3_vectors], axis=-1)/body3_features[:, :, np.newaxis]

        #for b3, f in zip(trimer_list, body3_features):
        #    print(b3[2].i, b3[2].j, b3[2].pos_i, b3[2].pos_j, b3[2].shift, f[2])
        #print(body3_features)
        #np.savetxt("b3.dat", body3_features, fmt="%8.2f")
    else:
        body3_features = []
        body3_gradients = []

    #print("BODY-3: ")
    #print(body3_mapping)
    #print(body3_vectors)
    #print(body3_features)
    #print("gradient: ", body3_gradients.shape)

    return body2_mapping, body2_features, body2_gradients, body3_mapping, body3_features, body3_gradients

if __name__ == "__main__":
    ...