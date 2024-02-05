#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp 
from scipy import optimize

import matplotlib.pyplot as plt

from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs

from itertools import permutations, product


def gaussian_kernel(x1, x2, delta=0.2, theta=0.5):
    """
        x1: shape (num_points, num_feature_dim)
        x2: shape (num_sparse_points, num_feature_dim)

        -> shape (num_points, num_sparse_points)

    """
    # shape (num_points, num_sparse_points, num_feature_dim)
    x_diff = x1[:, np.newaxis, :].repeat(x2.shape[0], axis=1) - x2[np.newaxis, :, :]

    return delta**2*np.exp(-np.sum(x_diff**2, axis=-1)/2./theta**2)

def gaussian_kernel_value_and_grad(x1, x2, delta=0.2, theta=0.5):
    """
    """
    # shape (num_points, num_sparse_points, num_feature_dim)
    x_diff = x1 - x2.T
    v = delta**2*np.exp(-x_diff**2/2./theta**2)

    # gradient with respect to x_diff
    g1 = v*(-x_diff/theta**2) 

    # gradient with respect to delta
    g2 = 2*v/delta

    # grad wrt to theta
    #g3 = v*(x_diff**2/theta**3)

    return (v, g1, g2)

def xxx_gaussian_kernel_value_and_grad(x1, x2, delta=0.2, theta=0.5):
    """
        x1: shape (num_points, num_feature_dim)
        x2: shape (num_sparse_points, num_feature_dim)

        v: shape (num_points, num_sparse_points)
        v: shape (num_points, num_sparse_points)
        ### g: shape (num_points, num_sparse_points, num_feature_dim)

    """
    #x_diff = x1 - x2
    #v = delta**2*np.exp(-x_diff**2/2./theta**2)

    # shape (num_points, num_sparse_points, num_feature_dim)
    x_diff = x1[:, np.newaxis, :].repeat(x2.shape[0], axis=1) - x2

    v = delta**2*np.exp(-np.sum(x_diff**2, axis=-1)/2./theta**2)

    # gradient with respect to x_diff
    #g = v*(-np.linalg.norm(x_diff, axis=-1)/theta**2) 

    num_feature_dim = x_diff.shape[-1]
    g = (
        np.repeat(v[:, :, np.newaxis], num_feature_dim, axis=-1) * 
        (-x_diff/theta**2)
    ).squeeze(axis=-1)

    return (v, g)


class DistanceDescriptor():

    def __init__(self) -> None:
        """"""

        return
    
def compute_body2_descriptor(frames, r_cut: float):
    """"""
    # -- get neighbours
    distance_mapping_list = []
    distances = []
    dis_derivs = []

    for i_frame, atoms in enumerate(frames):
        print(f"frame: {i_frame}")
        natoms = len(atoms)
        nl = NeighborList(
            cutoffs=[r_cut/2.]*natoms, skin=0.0, sorted=False,
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

    print(f"nfeatures: {len(distances)}")
    # distance descriptor, shape (num_distances, 1)
    distances = np.array(distances)[:, np.newaxis] 
    num_distances = distances.shape[0]
    print(distance_mapping_list)
    assert len(distance_mapping_list) == num_distances

    dis_derivs = np.array(dis_derivs)

    return distances, distance_mapping_list, dis_derivs
    
def compute_distance_kernal_matrices(
    delta, sigma2, nframes, natoms_list, y_data, 
    distances, dis_derivs, distance_mapping_list, sparse_points
):
    """"""
    # --
    natoms_tot = np.sum(natoms_list)
    num_sparse = sparse_points.shape[0]
    num_points = y_data.shape[0]

    # -- compute Kmm
    Kmm, _, Kmm_grad_wrt_delta = gaussian_kernel_value_and_grad(
        sparse_points, sparse_points, delta=delta
    )
    #print("Kmm_grad_wrt_delta: ")
    #print(Kmm_grad_wrt_delta)

    # -- compute Kmn
    #kernels = gaussian_kernel(distances, sparse_points.T)
    kernels, kernel_derivs, kernel_grads_wrt_delta = gaussian_kernel_value_and_grad(
        distances, sparse_points, delta=delta
    )
    #print(kernels)

    # --- derivatives
    #print("distance derivatives: ") # shape (num_distances, 6)
    #print(dis_derivs)
        
    #print("kernels: ")
    #print(kernels)

    #print("kernel derivatives: ") # shape (num_distances, num_sparse)
    #print(kernel_derivs)
        
    #print("kernel derivatives to cartesian: ") # shape (num_distances, 6, num_sparse)
    kernel_derivs = dis_derivs[:, :, np.newaxis].repeat(num_sparse, axis=2) * kernel_derivs[:, np.newaxis, :]
    #print(kernel_derivs)
    #print(kernel_derivs[0])

    #print(kernals)
    # --- group descriptors
    #print("===== group descriptor gradients =====")
    Knm_ene = np.zeros((nframes, num_sparse))
    Knm_ene_grad_wrt_delta = np.zeros((nframes, num_sparse))
    for i in range(nframes):
        for dis_loc, kernel, k_grad in zip(distance_mapping_list, kernels, kernel_grads_wrt_delta):
            if dis_loc[0] == i:
                Knm_ene[i, :] += kernel
                Knm_ene_grad_wrt_delta[i, :] += k_grad

    Knm_grad = np.zeros((natoms_tot*3, num_sparse))
    Knm_grad_grad_wrt_delta = np.zeros((natoms_tot*3, num_sparse))
    for dis_loc, kernel_grad in zip(distance_mapping_list, kernel_derivs):
        i = np.sum(natoms_list[:dis_loc[0]], dtype=int)*3 + dis_loc[1]*3
        j = np.sum(natoms_list[:dis_loc[0]], dtype=int)*3 + dis_loc[2]*3
        #print(i, j)
        Knm_grad[i:i+3, :] += kernel_grad[0:3, :]
        Knm_grad[j:j+3, :] += kernel_grad[3:6, :]
        Knm_grad_grad_wrt_delta[i:i+3, :] += kernel_grad[0:3, :]*2/delta
        Knm_grad_grad_wrt_delta[j:j+3, :] += kernel_grad[3:6, :]*2/delta
    Knm_frc = -Knm_grad

    #print("kernel on forces: ")
    #print(Knm_frc)

    # combine energy and force kernel
    Knm = np.vstack([Knm_ene, Knm_frc])

    return Kmm, Kmm_grad_wrt_delta, Knm, Knm_ene_grad_wrt_delta, Knm_grad_grad_wrt_delta

def compute_distance_marginal_likelihood(
    delta, sigma2, nframes, natoms_list, y_data, 
    distances, dis_derivs, distance_mapping_list, sparse_points
):
    num_points = y_data.shape[0]

    Kmm, Kmm_grad_wrt_delta, Knm, Knm_ene_grad_wrt_delta, Knm_grad_grad_wrt_delta = compute_distance_kernal_matrices(
        delta, sigma2, nframes, natoms_list, y_data, 
        distances, dis_derivs, distance_mapping_list, sparse_points
    )

    Kmm_inv = np.linalg.inv(Kmm)
    Knn = np.dot(Knm, np.dot(Kmm_inv, Knm.T)) + sigma2*np.eye(num_points)
    Knn_inv = np.linalg.inv(Knn)
    loss = -0.5*np.log(np.linalg.det(Knn)) - 0.5 * y_data.T @ Knn_inv @ y_data - num_points/2.*np.log(2*np.pi)

    # -- get gradients
    Knm_grad_wrt_delta = np.vstack([Knm_ene_grad_wrt_delta, -Knm_grad_grad_wrt_delta])
    Kmm_inv_grad_wrt_delta = -Kmm_inv @ Kmm_grad_wrt_delta @ Kmm_inv
    Ky = Knn_inv @ y_data

    Knn_grad_wrt_delta = (
        (Knm_grad_wrt_delta @ Kmm_inv + Knm @ Kmm_inv_grad_wrt_delta) @ Knm.T +
        Knm @ Kmm_inv @ Knm_grad_wrt_delta.T
    )

    loss_grad = 0.5*np.trace((Ky @ Ky.T - Knn_inv) @ Knn_grad_wrt_delta)

    return -loss, -loss_grad


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

        y_data = np.vstack([energies, forces])

        # - get atomic environments
        distances, distance_mapping_list, dis_derivs = compute_body2_descriptor(
            frames, self.r_cut
        )

        # - train
        # -- parameters
        sigma2 = 0.008
        delta = 0.2
        sparse_points = np.array([0.5, 1.0, 1.5, 2.0, 2.5])[:, np.newaxis]

        # -- compute coefficients
        Kmm, _, Knm, _, _ = compute_distance_kernal_matrices(
            delta, sigma2, nframes, natoms_list, y_data, 
            distances, dis_derivs, distance_mapping_list, sparse_points
        )
        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)

        loss = compute_distance_marginal_likelihood(
            delta, sigma2, nframes, natoms_list, y_data, 
            distances, dis_derivs, distance_mapping_list, sparse_points
        )
        print(f"LOSS: {loss}")

        res = optimize.minimize(
            compute_distance_marginal_likelihood, delta,
            args=(
                sigma2, nframes, natoms_list, y_data, 
                distances, dis_derivs, distance_mapping_list, sparse_points
            ), jac=True
        )
        print(res)

        # -- compute coefficients
        Kmm, _, Knm, _, _ = compute_distance_kernal_matrices(
            res.x, sigma2, nframes, natoms_list, y_data, 
            distances, dis_derivs, distance_mapping_list, sparse_points
        )
        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)

        return
    
    def _train_and_predict(self, nframes, sigma2, y_data, Kmm, Knm):
        """"""
        jitter = 1e-5*np.eye(Kmm.shape[0])
        inverseLamb = np.reciprocal(sigma2)*np.eye(y_data.shape[0])

        weights = (
            np.linalg.inv(Kmm + jitter + Knm.T@inverseLamb@Knm) @ Knm.T @ inverseLamb
        )
        weights = np.dot(weights, y_data)
        print("weights: ")
        print(weights)

        # - test
        print("===== TEST =====")
        #print("DFT: ")
        dft_data = y_data.flatten()
        #print("SGP: ")
        predictions = np.dot(Knm, weights)
        sgp_data  = predictions.flatten()
        print("ERR: ")
        ene_rmse = np.sqrt(np.sum((dft_data[:nframes] - sgp_data[:nframes])**2))
        print(f"ene_rmse: {ene_rmse}")
        frc_rmse = np.sqrt(np.sum((dft_data[nframes:] - sgp_data[nframes:])**2))
        print(f"frc_rmse: {frc_rmse}")

        return


if __name__ == "__main__":
    """"""
    sgp_trainer = SparseGaussianProcessTrainer()
    sgp_trainer.run()
    ...