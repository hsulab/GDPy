#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp 

import matplotlib.pyplot as plt

from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs

from itertools import permutations, product


def gaussian_kernel(x1, x2, delta=0.2, theta=0.5):
    """"""

    return delta**2*np.exp(-(x1-x2)**2/2./theta**2)

def gaussian_kernel_value_and_grad(x1, x2, delta=0.2, theta=0.5):
    """"""
    x_diff = x1 - x2

    v = delta**2*np.exp(-x_diff**2/2./theta**2)
    g = v*(-x_diff/theta**2) # gradient with respect to x_diff

    return (v, g)


class SparseGaussianProcessTrainer():

    def __init__(self) -> None:
        """"""
        self.r_cut = 4.0
        self.max_num_neigh = 3

        return
    
    def run(self, *args, **kwargs):
        """"""
        # - read dataset
        frames = read("./Cu4.xyz", ":1")
        nframes = len(frames)
        natoms_list = [len(a) for a in frames]
        natoms_tot = np.sum(natoms_list)

        energies = [a.get_potential_energy() for a in frames]
        energies = np.array(energies)[:, np.newaxis]

        forces = np.vstack([a.get_forces() for a in frames])
        forces = forces.flatten()[:, np.newaxis]
        print(f"force shape: {forces.shape}")

        # - get neighbours
        curr_start_atomic_index = 0
        distance_mapping_list = []

        distances = []
        dis_derivs = []
        for i_frame, atoms in enumerate(frames):
            print(f"frame: {i_frame}")
            natoms = len(atoms)
            nl = NeighborList(
                cutoffs=[self.r_cut/2.]*natoms, skin=0.0, sorted=False,
                self_interaction=False, bothways=True
            )
            nl.update(atoms)
            for i in range(natoms):
                nei_indices, nei_offsets = nl.get_neighbors(i)
                #print(i)
                #print(nei_indices, nei_offsets)
                for j, offset in zip(nei_indices, nei_offsets):
                    pos_i, pos_j = atoms.positions[i, :], atoms.positions[j, :]
                    pos_vec = pos_i - pos_j
                    distance = np.linalg.norm(
                        pos_vec + np.dot(offset, atoms.get_cell())
                    )
                    distances.append(distance)
                    dis_deriv = np.hstack([pos_vec/distance, -pos_vec/distance])
                    dis_derivs.append(dis_deriv)
                    distance_mapping_list.append((i_frame, i, j))
            curr_start_atomic_index += natoms

        print(f"nfeatures: {len(distances)}")
        # distance descriptor, shape (num_distances, 1)
        distances = np.array(distances)[:, np.newaxis] 
        num_distances = distances.shape[0]
        print(distance_mapping_list)
        assert len(distance_mapping_list) == num_distances

        dis_derivs = np.array(dis_derivs)

        # - train
        # -- compute Kmm
        sparse_points = np.array([0.5, 1.0, 1.5, 2.0, 2.5])[:, np.newaxis]
        num_sparse = sparse_points.shape[0]

        Kmm = gaussian_kernel(sparse_points, sparse_points.T)
        #print(Kmm)

        # -- compute Kmn
        #kernels = gaussian_kernel(distances, sparse_points.T)
        kernels, kernel_derivs = gaussian_kernel_value_and_grad(distances, sparse_points.T)

        # --- derivatives
        #print("distance derivatives: ") # shape (num_distances, 6)
        #print(dis_derivs)
        
        print("kernels: ")
        #print(kernels)

        print("kernel derivatives: ") # shape (num_distances, num_sparse)
        #print(kernel_derivs)
        
        print("kernel derivatives to cartesian: ") # shape (num_distances, 6, num_sparse)
        kernel_derivs = dis_derivs[:, :, np.newaxis].repeat(num_sparse, axis=2) * kernel_derivs[:, np.newaxis, :]
        #print(kernel_derivs)
        #print(kernel_derivs[0])

        #print(kernals)
        # --- group descriptors
        print("===== group descriptor gradients =====")
        Knm_ene = np.zeros((nframes, num_sparse))
        for i in range(nframes):
            for dis_loc, kernel in zip(distance_mapping_list, kernels):
                if dis_loc[0] == i:
                    Knm_ene[i, :] += kernel

        Knm_grad = np.zeros((natoms_tot*3, num_sparse))
        for dis_loc, kernel_grad in zip(distance_mapping_list, kernel_derivs):
            i = np.sum(natoms_list[:dis_loc[0]], dtype=int)*3 + dis_loc[1]*3
            j = np.sum(natoms_list[:dis_loc[0]], dtype=int)*3 + dis_loc[2]*3
            #print(i, j)
            Knm_grad[i:i+3, :] += kernel_grad[0:3, :]
            Knm_grad[j:j+3, :] += kernel_grad[3:6, :]
        Knm_frc = -Knm_grad

        print("kernel on forces: ")
        print(Knm_frc)

        # combine energy and force kernel
        Kmn = np.vstack([Knm_ene, Knm_frc]).T
        
        # -- compute coefficients
        Y_data = np.vstack([energies, forces])

        jitter = 1e-5*np.eye(Kmm.shape[0])
        inverseLamb = np.reciprocal(0.008)*np.eye(Y_data.shape[0])

        weights = np.dot(
            np.dot(
                np.linalg.inv(Kmm + jitter + np.dot(np.dot(Kmn, inverseLamb), Kmn.T)), 
                Kmn
            ),
            inverseLamb
        )
        weights = np.dot(weights, Y_data)
        #print("weights: ")
        #print(weights)

        # - test
        print("===== TEST =====")
        #print("DFT: ")
        dft_data = Y_data.flatten()
        #print("SGP: ")
        predictions = np.dot(Kmn.T, weights)
        sgp_data  = predictions.flatten()
        print("ERR: ")
        ene_rmse = np.sqrt(np.sum((dft_data[:nframes] - sgp_data[:nframes])**2))
        print(f"ene_rmse: {ene_rmse}")
        frc_rmse = np.sqrt(np.sum((dft_data[nframes:] - sgp_data[nframes:])**2))
        print(f"frc_rmse: {frc_rmse}")

        return
    
    def _compute_Kmn(self, kernel, sparse_points, X_data):
        """"""
        sparse_points = np.array([0.5, 1.0, 1.5, 2.0, 2.5])[:, np.newaxis]
        print(f"sparse: {sparse_points}")
        num_sparse = sparse_points.shape[0]
        num_data = X_data.shape[0]

        Kmn = np.zeros((num_sparse, num_data))
        for i in range(num_data):
            atomic_env = X_data[i]
            #print(atomic_env)
            #print(atomic_env - sparse_points)

        return



if __name__ == "__main__":
    """"""
    sgp_trainer = SparseGaussianProcessTrainer()
    sgp_trainer.run()
    ...