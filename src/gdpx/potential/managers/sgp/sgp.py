#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import itertools

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

    # gradient wrt x1
    g0 = v*(-x_diff/theta**2) # <---

    # grad wrt delta
    g2 = 2*v/delta

    # grad wrt to theta
    g3 = v*(x_diff**2/theta**3)

    # gradient wrt x1 wrt theta
    #g0theta = ((2*x_diff/theta**3) + (-x_diff**3/theta**5))/(-x_diff/theta**2)
    #g0theta = ((-2/theta) + (x_diff**2/theta**3))

    return (v, g0, g2, g3)

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
    
def compute_body2_descriptor(frames, r_cut: float):
    """"""
    # -- get neighbours
    distance_mapping_list = []
    distances = []
    dis_derivs = []

    for i_frame, atoms in enumerate(frames):
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

    # distance descriptor, shape (num_distances, 1)
    distances = np.array(distances)[:, np.newaxis] 
    num_distances = distances.shape[0]
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
    Kmm, _, Kmm_grad_wrt_delta, _ = gaussian_kernel_value_and_grad(
        sparse_points, sparse_points, delta=delta
    )
    #print("Kmm_grad_wrt_delta: ")
    #print(Kmm_grad_wrt_delta)

    # -- compute Kmn
    #kernels = gaussian_kernel(distances, sparse_points.T)
    kernels, kernel_derivs, kernel_grads_wrt_delta, _ = gaussian_kernel_value_and_grad(
        distances, sparse_points, delta=delta
    )
    #print("b2_kernels: ")
    #print(kernels)

    #print("kernel derivatives: ") # shape (num_distances, num_sparse)
    #print(kernel_derivs)

    # --- derivatives
    #print("distance derivatives: ") # shape (num_distances, 6)
    #print(dis_derivs)
        
    #print("kernels: ")
    #print(kernels)

    #print("kernel derivatives to cartesian: ") # shape (num_distances, 6, num_sparse)
    kernel_derivs = dis_derivs[:, :, np.newaxis].repeat(num_sparse, axis=2) * kernel_derivs[:, np.newaxis, :]
    #print(kernel_derivs[0])
    #print(kernel_derivs.shape)
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

    #print("Knm: ")
    #print(Knm)

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


DimerData = collections.namedtuple("DimerData", ["fi", "i", "j", "pos_i", "pos_j", "shift"])

def compute_body3_descriptor(frames, r_cut: float):
    """"""
    # -- get neighbours
    distance_mapping_list = []
    distance_vectors = []
    dis_derivs = []

    dimer_list = []

    curr_atomic_index = 0
    for i_frame, atoms in enumerate(frames):
        #print(f"frame: {i_frame}")
        natoms = len(atoms)
        nl = NeighborList(
            cutoffs=[r_cut/2.]*natoms, skin=0.0, sorted=False,
            self_interaction=False, bothways=True
        )
        nl.update(atoms)
        for i in range(natoms):
            nei_indices, nei_offsets = nl.get_neighbors(i)
            for j, offset in zip(nei_indices, nei_offsets):
                pos_i, pos_j = atoms.positions[i, :], atoms.positions[j, :]
                shift = -np.dot(offset, atoms.get_cell()) # take negative
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
    print("BODY-2: ")
    print(body2_mapping)
    print(body2_features)
    #print(body2_gradients)

    # - find 3-body
    trimer_list = []
    for k, v in itertools.groupby(dimer_list, key=lambda x: x.i):
        for pair0, pair1 in itertools.combinations(v, 2):
            pos_i, pos_j = pair0.pos_j + pair0.shift, pair1.pos_j + pair1.shift
            distance = np.linalg.norm(pos_i - pos_j)
            if distance <= r_cut:
                pair2 = DimerData(
                    fi=pair0.fi,
                    i=pair0.j, j=pair1.j,
                    pos_i=pos_i, pos_j=pos_j, shift=np.zeros(3)
                )
                trimer_list.append((pair0, pair1, pair2))
    body3_mapping, body3_vectors = [], []
    for pairs in trimer_list:
        body3_mapping.append([pairs[0].fi, pairs[0].i, pairs[0].j, pairs[1].j])
        body3_vectors.append([p.pos_i-(p.pos_j+shift) for p in pairs]) 
    body3_vectors = np.array(body3_vectors) # shape (n_trimers, 3, 3)
    body3_features = np.linalg.norm(body3_vectors, axis=2) # shape (n_trimers, 3)
    body3_gradients = np.concatenate([body3_vectors, -body3_vectors], axis=-1)/body3_features[:, :, np.newaxis]

    print("BODY-3: ")
    print(body3_mapping)
    #print(body3_vectors)
    print(body3_features)
    print("gradient: ", body3_gradients.shape)

    return body2_mapping, body2_features, body2_gradients, body3_mapping, body3_features, body3_gradients

def gaussian_kernel_value_and_grad_body3(x1, x2, delta=0.2, theta=0.5):
    """
        x1: shape (num_points, num_feature_dim)
        x2: shape (num_sparse_points, num_feature_dim)

        v: shape (num_points, num_sparse_points)
        v: shape (num_points, num_sparse_points)
        ### g: shape (num_points, num_sparse_points, num_feature_dim)

    """
    # shape (num_points, num_sparse_points, num_feature_dim)
    x_diff = x1[:, np.newaxis, :].repeat(x2.shape[0], axis=1) - x2

    v = delta**2*np.exp(-np.linalg.norm(x_diff, axis=-1)**2/2./theta**2)

    # gradient wrt x1
    num_feature_dim = x_diff.shape[-1]
    g0 = (
        np.repeat(v[:, :, np.newaxis], num_feature_dim, axis=-1) * 
        (-x_diff/theta**2)
    )

    return (v, g0)

def compute_body3_kernel_matrices(
    delta, nframes, natoms_tot, 
    body3_features, body3_gradients, body3_mapping, sparse_body3_features
):
    """"""
    num_sparse = sparse_body3_features.shape[0]


    # -- consider permutations...
    b3_kernels = np.zeros((body3_features.shape[0], num_sparse))
    b3_kernel_grad_x1 = np.zeros((body3_features.shape[0], num_sparse, 3))
    for p in itertools.permutations(range(3), 3):
        k, k_g_x1 = gaussian_kernel_value_and_grad_body3(
            body3_features, sparse_body3_features[:, p], delta=delta
        )
        b3_kernels += k
        b3_kernel_grad_x1 += k_g_x1

    print("CONSTRUCT B3 KNM: ")
    #b3_kernels, b3_kernel_grad_x1 = gaussian_kernel_value_and_grad_body3(
    #    body3_features, sparse_body3_features[:, (0, 1, 2)], delta=delta
    #)

    # --- group descriptors
    Knm_ene = np.zeros((nframes, num_sparse))
    for loc, kernel in zip(body3_mapping, b3_kernels):
        fi, i, j, k = loc
        Knm_ene[fi, :] += kernel

    Knm_grad = np.zeros((natoms_tot*3, num_sparse))
    for loc, b3_grad, k_grad in zip(body3_mapping, body3_gradients, b3_kernel_grad_x1):
        fi, i, j, k = loc
        # b3_grad (3, 6) k_grad (num_sparse, 3) -> (num_sparse, 3, 6)
        curr_grad = np.repeat(k_grad[:, :, np.newaxis], 6, axis=2) * b3_grad
        Knm_grad[i*3:i*3+3, :] += curr_grad[:, 0, 0:3].T
        Knm_grad[j*3:j*3+3, :] += curr_grad[:, 0, 3:6].T
        Knm_grad[i*3:i*3+3, :] += curr_grad[:, 1, 0:3].T
        Knm_grad[k*3:k*3+3, :] += curr_grad[:, 1, 3:6].T
        Knm_grad[j*3:j*3+3, :] += curr_grad[:, 2, 0:3].T
        Knm_grad[k*3:k*3+3, :] += curr_grad[:, 2, 3:6].T
    Knm_frc = -Knm_grad

    Knm = np.vstack([Knm_ene, Knm_frc])
    print("Knm shape: ")
    print(Knm.shape)
    #print(Knm)

    # --- construct Kmm
    Kmm, _ = gaussian_kernel_value_and_grad_body3(
        sparse_body3_features, sparse_body3_features, delta=delta
    )
    print("Kmm shape: ")
    print(Kmm.shape)

    return Kmm, Knm

def compute_body2_kernel_matrices(
    delta, theta, nframes, natoms_tot, 
    body2_features, body2_gradients, body2_mapping, sparse_body2_features
):
    """"""
    num_sparse = sparse_body2_features.shape[0]

    # - 
    b2_kernels, b2_kernel_grad_x1, b2_kg_wrt_delta, b2_kg_wrt_theta = gaussian_kernel_value_and_grad(
        body2_features, sparse_body2_features, delta=delta, theta=theta
    )
    b2_kernels = b2_kernels # (num_b2, num_sparse)
    b2_kernel_grad_x1 = b2_kernel_grad_x1[:, :, np.newaxis] # (num_b2, num_sparse, 1)

    b2_kg_wrt_delta = b2_kg_wrt_delta[:, :, np.newaxis] # (num_b2, num_sparse, 1)
    b2_kg_wrt_theta = b2_kg_wrt_theta[:, :, np.newaxis] # (num_b2, num_sparse, 1)

    # --- group descriptors
    #print("CONSTRUCT B2 KNM: ")
    Knm_ene = np.zeros((nframes, num_sparse))
    Knm_ene_gdelta = np.zeros((nframes, num_sparse))
    Knm_ene_gtheta = np.zeros((nframes, num_sparse))
    for loc, kernel, k_gdelta, k_gtheta in zip(body2_mapping, b2_kernels, b2_kg_wrt_delta, b2_kg_wrt_theta):
        fi, i, j = loc
        Knm_ene[fi, :] += kernel
        Knm_ene_gdelta[fi, :] += k_gdelta.squeeze()
        Knm_ene_gtheta[fi, :] += k_gtheta.squeeze() # PASS

    Knm_grad = np.zeros((natoms_tot*3, num_sparse))
    Knm_gcart_gdelta = np.zeros((natoms_tot*3, num_sparse))
    Knm_gcart_gtheta = np.zeros((natoms_tot*3, num_sparse))
    b2ks_gx1_gtheta_coef = (-2./theta + (body2_features-sparse_body2_features.T)**2/theta**3)[:, :, np.newaxis]
    for loc, b2_grad, k_grad, b2k_gx1gtheta in zip(
        body2_mapping, body2_gradients, b2_kernel_grad_x1, b2ks_gx1_gtheta_coef
    ):
        fi, i, j = loc
        # b2_grad (1, 6) k_grad (num_sparse, 1) -> (num_sparse, 1, 6)
        curr_grad = np.repeat(k_grad[:, :, np.newaxis], 6, axis=2) * b2_grad
        Knm_grad[i*3:i*3+3, :] += curr_grad[:, 0, 0:3].T
        Knm_grad[j*3:j*3+3, :] += curr_grad[:, 0, 3:6].T
        # --
        Knm_gcart_gdelta[i*3:i*3+3, :] += curr_grad[:, 0, 0:3].T*2./delta
        Knm_gcart_gdelta[j*3:j*3+3, :] += curr_grad[:, 0, 3:6].T*2./delta
        # --
        Knm_gcart_gtheta[i*3:i*3+3, :] += (curr_grad[:, 0, 0:3]*b2k_gx1gtheta).T
        Knm_gcart_gtheta[j*3:j*3+3, :] += (curr_grad[:, 0, 3:6]*b2k_gx1gtheta).T
    Knm_frc = -Knm_grad

    Knm = np.vstack([Knm_ene, Knm_frc])
    #print("Knm shape: ")
    #print(Knm.shape)

    # ---

    # --- construct Kmm
    Kmm, _, Kmm_gdelta, Kmm_gtheta  = gaussian_kernel_value_and_grad(
        sparse_body2_features, sparse_body2_features, delta=delta, theta=theta
    )
    #print("Kmm shape: ", Kmm.shape)

    return Kmm, Knm, Kmm_gdelta, Kmm_gtheta, Knm_ene_gdelta, Knm_ene_gtheta, Knm_gcart_gdelta, Knm_gcart_gtheta

def compute_b2b3_marginal_likelihood(
    params, sigma2, y_data, nframes, natoms_tot, 
    body2_features, body2_gradients, body2_mapping, sparse_body2_features
):
    delta, theta = params

    num_points = y_data.shape[0]

    Kmm, Knm, Kmm_gdelta, Kmm_gtheta, Knm_ene_gdelta, Knm_ene_gtheta, Knm_gcart_gdelta, Knm_gcart_gtheta = compute_body2_kernel_matrices(
        delta, theta, nframes, natoms_tot, 
        body2_features, body2_gradients, body2_mapping, sparse_body2_features
    )

    Kmm_inv = np.linalg.inv(Kmm)
    Knn = np.dot(Knm, np.dot(Kmm_inv, Knm.T)) + sigma2*np.eye(num_points)
    Knn_inv = np.linalg.inv(Knn)
    loss = -0.5*np.log(np.linalg.det(Knn)) - 0.5 * y_data.T @ Knn_inv @ y_data - num_points/2.*np.log(2*np.pi)

    # -- get gradients
    Knm_grad_wrt_delta = np.vstack([Knm_ene_gdelta, -Knm_gcart_gdelta])
    Kmm_inv_grad_wrt_delta = -Kmm_inv @ Kmm_gdelta @ Kmm_inv

    Knn_grad_wrt_delta = (
        (Knm_grad_wrt_delta @ Kmm_inv + Knm @ Kmm_inv_grad_wrt_delta) @ Knm.T +
        Knm @ Kmm_inv @ Knm_grad_wrt_delta.T
    )

    Knm_grad_wrt_theta = np.vstack([Knm_ene_gtheta, -Knm_gcart_gtheta])
    Kmm_inv_grad_wrt_theta = -Kmm_inv @ Kmm_gtheta @ Kmm_inv

    Knn_grad_wrt_theta = (
        (Knm_grad_wrt_theta @ Kmm_inv + Knm @ Kmm_inv_grad_wrt_theta) @ Knm.T +
        Knm @ Kmm_inv @ Knm_grad_wrt_theta.T
    )

    Ky = Knn_inv @ y_data

    loss_gdelta = 0.5*np.trace((Ky @ Ky.T - Knn_inv) @ Knn_grad_wrt_delta)
    loss_gtheta = 0.5*np.trace((Ky @ Ky.T - Knn_inv) @ Knn_grad_wrt_theta)

    return -loss, [-loss_gdelta, -loss_gtheta]
    #return -loss, -loss_gdelta


class SparseGaussianProcessTrainer():

    def __init__(self) -> None:
        """"""
        self.r_cut = 6.8
        self.max_num_neigh = 3

        return
    
    def _prepare_dataset(self, ):
        """"""
        # - read dataset
        frames = read("./Cu4.xyz", ":1")

        energies = [a.get_potential_energy() for a in frames]
        energies = np.array(energies)[:, np.newaxis]

        forces = np.vstack([a.get_forces() for a in frames])
        forces = forces.flatten()[:, np.newaxis]
        print(f"force shape: {forces.shape}")

        y_data = np.vstack([energies, forces])

        return frames, y_data
    
    def run(self, *args, **kwargs):
        """"""
        # - read dataset
        frames, y_data = self._prepare_dataset()
        nframes = len(frames)
        natoms_list = [len(a) for a in frames]
        natoms_tot = np.sum(natoms_list)

        num_points = nframes + natoms_tot*3

        # - get atomic environments
        (
            body2_mapping, body2_features, body2_gradients, body3_mapping, body3_features, body3_gradients
        ) = compute_body3_descriptor(frames, self.r_cut)

        # - construct matrix
        # -- parameters
        sigma2 = 0.008
        delta, theta = 0.2, 0.5

        # --- body2
        sparse_body2_features = np.array([0.5, 1.0, 1.5, 2.0, 2.5])[:, np.newaxis]
        Kmm_b2, Knm_b2, _, _, _, _, _, _ = compute_body2_kernel_matrices(
            delta, theta, nframes, natoms_tot, 
            body2_features, body2_gradients, body2_mapping, sparse_body2_features
        )
        Kmm = Kmm_b2
        Knm = Knm_b2

        # - train
        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)

        ret = compute_b2b3_marginal_likelihood(
            (delta, theta), sigma2, y_data, nframes, natoms_tot, 
            body2_features, body2_gradients, body2_mapping, sparse_body2_features
        )
        print(ret)

        res = optimize.minimize(
            compute_b2b3_marginal_likelihood, [delta, theta],
            args=(
                sigma2, y_data, nframes, natoms_tot, 
                body2_features, body2_gradients, body2_mapping, sparse_body2_features
            ), jac=True
        )
        print("OPT INFO: ")
        print(res)
        delta, theta = res.x
        Kmm_b2, Knm_b2, _, _, _, _, _, _ = compute_body2_kernel_matrices(
            delta, theta, nframes, natoms_tot, 
            body2_features, body2_gradients, body2_mapping, sparse_body2_features
        )
        Kmm = Kmm_b2
        Knm = Knm_b2

        # - train
        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)
        
        # --- body3
        #sparse_body3_features = np.array(
        #    [
        #        [1, 2, 3],
        #        [1.5, 2.5, 3.5],
        #        [2, 3, 4],
        #        [2.5, 3.5, 4.5],
        #        [4, 5, 6],
        #    ]
        #)
        #Kmm_b3, Knm_b3 = compute_body3_kernel_matrices(
        #    delta, nframes, natoms_tot, 
        #    body3_features, body3_gradients, body3_mapping, sparse_body3_features
        #)

        #Kmm = Kmm_b3
        #Knm = Knm_b3

        # -- combine matrices
        #num_b2_sparse = sparse_body2_features.shape[0]
        #num_b3_sparse = sparse_body3_features.shape[0]

        #Kmm = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
        #Kmm[:num_b2_sparse, :num_b2_sparse] = Kmm_b2
        #Kmm[num_b2_sparse:, num_b2_sparse:] = Kmm_b3

        #Knm = np.zeros((num_points, num_b2_sparse+num_b3_sparse))
        #Knm[:, :num_b2_sparse] = Knm_b2
        #Knm[:, num_b2_sparse:] = Knm_b3

        return
    
    def run2(self, *args, **kwargs):
        """"""
        # - read dataset
        frames, y_data = self._prepare_dataset()
        nframes = len(frames)
        natoms_list = [len(a) for a in frames]
        natoms_tot = np.sum(natoms_list)

        # - get atomic environments
        distances, distance_mapping_list, dis_derivs = compute_body2_descriptor(
            frames, self.r_cut
        )
        print("BODY-2 FEATURE GRADIENTS W.R.T CARTESIAN: ")
        print(dis_derivs)

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

        # - train
        res = optimize.minimize(
            compute_distance_marginal_likelihood, delta,
            args=(
                sigma2, nframes, natoms_list, y_data, 
                distances, dis_derivs, distance_mapping_list, sparse_points
            ), jac=True
        )
        print("OPT INFO: ")
        print(res)

        ## -- compute coefficients
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
    print("!!!!! BODY-2 !!!!!") 
    sgp_trainer.run2()
    print("!!!!! BODY-2+3 !!!!!") 
    sgp_trainer.run()
    ...