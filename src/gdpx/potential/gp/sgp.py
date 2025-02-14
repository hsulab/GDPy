#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import itertools
from typing import List

import numpy as np
import scipy as sp 
from scipy import optimize

import matplotlib.pyplot as plt

from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs

from itertools import permutations, product

from gdpx.potential.trainer import AbstractTrainer
from .gptools import compute_body3_descriptor

def switch_function_value_and_grad(r, r_cut):
    """Switch function that smoothes distance to cutoff.
    """

    return (r-r_cut)**2, 2*(r-r_cut)

#def switch_function_value_and_grad(r, r_cut):
#    """Switch function that smoothes distance to cutoff.
#    """
#
#    return np.array([[1.]]), np.array([[0.]])

def switch_function_value_and_grad_b3(r, r_cut):
    """Switch function that smoothes distance to cutoff.
    """
    r_diff = (r-r_cut)
    v = np.prod(r_diff**2, axis=1)[:, np.newaxis]

    return v, 2./r_diff*v

#def switch_function_value_and_grad_b3(r, r_cut):
#    """Switch function that smoothes distance to cutoff.
#    """
#
#    return np.array([[1.]]), np.array([[0.]])

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
    #x_diff = np.linalg.norm((x1 - x2.T)[:, :, np.newaxis], axis=-1)
    v = delta**2*np.exp(-x_diff**2/2./theta**2)

    # gradient wrt x1
    g0 = v*(-x_diff/theta**2) # <---

    # grad wrt delta
    g2 = 2*v/delta

    # grad wrt theta
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
    Knn = np.dot(Knm, np.dot(Kmm_inv, Knm.T)) + sigma2
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

# --- BODY-3 ---

def gaussian_kernel_value_and_grad_body3(x1, x2, delta=0.2, theta=0.5):
    """
        x1: shape (num_points, num_feature_dim)
        x2: shape (num_sparse_points, num_feature_dim)

        v: shape (num_points, num_sparse_points)
        v: shape (num_points, num_sparse_points)
        ### g: shape (num_points, num_sparse_points, num_feature_dim)

    """
    # shape (num_points, num_sparse, num_feature_dim)
    x_diff = x1[:, np.newaxis, :].repeat(x2.shape[0], axis=1) - x2
    x_norm = np.linalg.norm(x_diff, axis=-1)

    # shape (num_points, num_sparse)
    v = delta**2*np.exp(-x_norm**2/2./theta**2)

    # gradient wrt x1
    num_feature_dim = x_diff.shape[-1]
    rv = np.repeat(v[:, :, np.newaxis], num_feature_dim, axis=-1)
    g0 = rv*(-x_diff/theta**2)

    # grad wrt delta
    g2 = v*2./delta

    # grad wrt theta
    g3 = v*(x_norm**2/theta**3)

    return (v, g0, g2, g3)


def compute_body3_kernel_matrices(
    delta, theta, r_cut, nframes, natoms_tot, 
    b3_features, body3_gradients, body3_mapping, sparse_body3_features
):
    """"""
    num_sparse = sparse_body3_features.shape[0]

    # NOTE: cutoff switch function on sparse_points
    #sparse_b3s, _ = switch_function_value_and_grad_b3(sparse_body3_features, r_cut)

    # -- consider permutations...
    num_b3 = b3_features.shape[0]
    b3_kernels = np.zeros((b3_features.shape[0], num_sparse))
    b3_kernel_grad_x1 = np.zeros((b3_features.shape[0], num_sparse, 3))
    b3ks_gdelta = np.zeros((num_b3, num_sparse))
    b3ks_gtheta = np.zeros((num_b3, num_sparse))
    #b3ks_gx1_gtheta_coef = np.zeros((num_b3, num_sparse))
    b3ks_gx1_gtheta = np.zeros((num_b3, num_sparse, 3))

    b3s, b3sg = switch_function_value_and_grad_b3(b3_features, r_cut)
    #b3s = b3s*sparse_b3s.T # shape (num_b3, num_sparse)
    #b3sg = b3sg[:, np.newaxis, :]*sparse_b3s.T[:, :, np.newaxis] # shape (num_b3, 1, 3)
    b3s = b3s # shape (num_b3, 1)
    b3sg = b3sg[:, np.newaxis, :] # shape (num_b3, 1, 3)

    for p in itertools.permutations(range(3), 3):
    #for p in [(0, 1, 2)]:
        _k, _k_g_x1, k_gdelta, k_gtheta = gaussian_kernel_value_and_grad_body3(
            b3_features, sparse_body3_features[:, p], delta=delta, theta=theta
        )
        # --
        b3_kernels += _k*b3s
        b3_kernel_grad_x1 += _k_g_x1*b3s[:, :, np.newaxis] + np.repeat(_k[:, :, np.newaxis], 3, axis=2)*b3sg
        b3ks_gdelta += k_gdelta*b3s
        b3ks_gtheta += k_gtheta*b3s
        # --
        x_norm = np.linalg.norm(
            b3_features[:, np.newaxis, :].repeat(num_sparse, axis=1)-sparse_body3_features[:, p], 
            axis=-1
        )[:, :, np.newaxis]
        #b3ks_gx1_gtheta_coef += (-2./theta + x_norm**2/theta**3)
        b3ks_gx1_gtheta += (-2./theta*_k_g_x1*b3s[:, :, np.newaxis] + x_norm**2/theta**3*b3_kernel_grad_x1)

    # --- group descriptors
    Knm_ene = np.zeros((nframes, num_sparse))
    Knm_ene_gdelta = np.zeros((nframes, num_sparse))
    Knm_ene_gtheta = np.zeros((nframes, num_sparse))
    for loc, kernel, k_gdelta, k_gtheta in zip(body3_mapping, b3_kernels, b3ks_gdelta, b3ks_gtheta):
        fi, i, j, k = loc
        Knm_ene[fi, :] += kernel          # PASS
        Knm_ene_gdelta[fi, :] += k_gdelta # PASS
        Knm_ene_gtheta[fi, :] += k_gtheta # PASS

    Knm_grad = np.zeros((natoms_tot*3, num_sparse))
    Knm_gcart_gdelta = np.zeros((natoms_tot*3, num_sparse))
    Knm_gcart_gtheta = np.zeros((natoms_tot*3, num_sparse))
    for loc, b3_grad, k_grad, k_gx1gtheta in zip(
        body3_mapping, body3_gradients, b3_kernel_grad_x1, b3ks_gx1_gtheta
    ):
        fi, i, j, k = loc
        # b3_grad (3, 6) k_grad (num_sparse, 3) -> (num_sparse, 3, 6)
        curr_grad = np.repeat(k_grad[:, :, np.newaxis], 6, axis=2) * b3_grad
        Knm_grad[i*3:i*3+3, :] += curr_grad[:, 0, 0:3].T
        Knm_grad[j*3:j*3+3, :] += curr_grad[:, 0, 3:6].T
        Knm_grad[i*3:i*3+3, :] += curr_grad[:, 1, 0:3].T
        Knm_grad[k*3:k*3+3, :] += curr_grad[:, 1, 3:6].T
        Knm_grad[j*3:j*3+3, :] += curr_grad[:, 2, 0:3].T
        Knm_grad[k*3:k*3+3, :] += curr_grad[:, 2, 3:6].T
        # -- PASS
        Knm_gcart_gdelta[i*3:i*3+3, :] += curr_grad[:, 0, 0:3].T*2./delta
        Knm_gcart_gdelta[j*3:j*3+3, :] += curr_grad[:, 0, 3:6].T*2./delta
        Knm_gcart_gdelta[i*3:i*3+3, :] += curr_grad[:, 1, 0:3].T*2./delta
        Knm_gcart_gdelta[k*3:k*3+3, :] += curr_grad[:, 1, 3:6].T*2./delta
        Knm_gcart_gdelta[j*3:j*3+3, :] += curr_grad[:, 2, 0:3].T*2./delta
        Knm_gcart_gdelta[k*3:k*3+3, :] += curr_grad[:, 2, 3:6].T*2./delta
        # -- 
        #Knm_gcart_gtheta[i*3:i*3+3, :] += (curr_grad[:, 0, 0:3]*k_gx1gtheta).T
        #Knm_gcart_gtheta[j*3:j*3+3, :] += (curr_grad[:, 0, 3:6]*k_gx1gtheta).T
        #Knm_gcart_gtheta[i*3:i*3+3, :] += (curr_grad[:, 1, 0:3]*k_gx1gtheta).T
        #Knm_gcart_gtheta[k*3:k*3+3, :] += (curr_grad[:, 1, 3:6]*k_gx1gtheta).T
        #Knm_gcart_gtheta[j*3:j*3+3, :] += (curr_grad[:, 2, 0:3]*k_gx1gtheta).T
        #Knm_gcart_gtheta[k*3:k*3+3, :] += (curr_grad[:, 2, 3:6]*k_gx1gtheta).T
        # -- b3_grad (3, 6) k_gx1gtheta (5,3) -> (3, 5, 6)
        curr_grad_theta = b3_grad[:, np.newaxis, :]*k_gx1gtheta.T[:, :, np.newaxis]
        Knm_gcart_gtheta[i*3:i*3+3, :] += curr_grad_theta[0, :, 0:3].T
        Knm_gcart_gtheta[j*3:j*3+3, :] += curr_grad_theta[0, :, 3:6].T
        Knm_gcart_gtheta[i*3:i*3+3, :] += curr_grad_theta[1, :, 0:3].T
        Knm_gcart_gtheta[k*3:k*3+3, :] += curr_grad_theta[1, :, 3:6].T
        Knm_gcart_gtheta[j*3:j*3+3, :] += curr_grad_theta[2, :, 0:3].T
        Knm_gcart_gtheta[k*3:k*3+3, :] += curr_grad_theta[2, :, 3:6].T
    Knm_frc = -Knm_grad

    Knm = np.vstack([Knm_ene, Knm_frc])
    #print("Knm shape: ")
    #print(Knm)
    #print(Knm.shape)
    #print(Knm_ene_gdelta)
    #print(Knm_gcart_gdelta)
    #print(Knm_gcart_gdelta.shape)
    #print(Knm_ene_gtheta)
    #print(Knm_gcart_gtheta)
    #print(Knm_gcart_gtheta.shape)

    # --- construct Kmm
    Kmm = np.zeros((num_sparse, num_sparse))
    Kmm_gdelta = np.zeros((num_sparse, num_sparse))
    Kmm_gtheta = np.zeros((num_sparse, num_sparse))
    # TODO: permutations?
    for p in itertools.permutations(range(3), 3):
    #for p in [(0, 1, 2)]:
        curr_Kmm, _, curr_Kmm_gdelta, curr_Kmm_gtheta = gaussian_kernel_value_and_grad_body3(
            sparse_body3_features, sparse_body3_features[:, p], delta=delta, theta=theta
        )
        Kmm += curr_Kmm
        Kmm_gdelta += curr_Kmm_gdelta
        Kmm_gtheta += curr_Kmm_gtheta

    # TODO: add cutoff function?
    #cutoff_sparse = sparse_b3s*sparse_b3s.T
    #Kmm = Kmm*cutoff_sparse

    #print("Kmm shape: ")
    #print(Kmm.shape)
    #print(Kmm)
    #exit()

    return Kmm, Knm, Kmm_gdelta, Kmm_gtheta, Knm_ene_gdelta, Knm_ene_gtheta, Knm_gcart_gdelta, Knm_gcart_gtheta

def compute_body2_kernel_matrices(
    delta, theta, r_cut, nframes, natoms_tot, 
    body2_features, body2_gradients, body2_mapping, sparse_body2_features
):
    """"""
    #print(f"r_cut: {r_cut}")
    num_sparse = sparse_body2_features.shape[0]

    #sparse_b2, _ = switch_function_value_and_grad(sparse_body2_features, r_cut)

    # - 
    _b2_kernels, _b2_kernel_grad_x1, b2_kg_wrt_delta, b2_kg_wrt_theta = gaussian_kernel_value_and_grad(
        body2_features, sparse_body2_features, delta=delta, theta=theta
    )
    #print(_b2_kernels.shape)
    #print(_b2_kernel_grad_x1.shape)
    b2s, b2sg = switch_function_value_and_grad(body2_features, r_cut)
    #b2s = b2s*sparse_b2.T # (num_b2, num_sparse)
    #b2sg = b2sg*sparse_b2.T # (num_b2, num_sparse)
    b2s = b2s # (num_b2, num_sparse)
    b2sg = b2sg # (num_b2, num_sparse)

    b2_kernels = _b2_kernels*b2s # (num_b2, num_sparse)
    b2_kernel_grad_x1 = (_b2_kernel_grad_x1*b2s+_b2_kernels*b2sg)[:, :, np.newaxis] # (num_b2, num_sparse, 1)
    #print(b2_kernel_grad_x1)
    #print(b2_kernel_grad_x1.shape)

    b2_kg_wrt_delta = (b2_kg_wrt_delta*b2s)[:, :, np.newaxis] # (num_b2, num_sparse, 1)
    b2_kg_wrt_theta = (b2_kg_wrt_theta*b2s)[:, :, np.newaxis] # (num_b2, num_sparse, 1)
    print(_b2_kernels)
    print(_b2_kernel_grad_x1)
    print(body2_gradients)

    # --- group descriptors
    #print("CONSTRUCT B2 KNM: ")
    Knm_ene = np.zeros((nframes, num_sparse))
    Knm_ene_gdelta = np.zeros((nframes, num_sparse))
    Knm_ene_gtheta = np.zeros((nframes, num_sparse))
    for loc, kernel, k_gdelta, k_gtheta in zip(body2_mapping, b2_kernels, b2_kg_wrt_delta, b2_kg_wrt_theta):
        fi, i, j = loc
        Knm_ene[fi, :] += kernel
        Knm_ene_gdelta[fi, :] += k_gdelta.squeeze() # PASS
        Knm_ene_gtheta[fi, :] += k_gtheta.squeeze() # PASS

    Knm_grad = np.zeros((natoms_tot*3, num_sparse))
    Knm_gcart_gdelta = np.zeros((natoms_tot*3, num_sparse))
    Knm_gcart_gtheta = np.zeros((natoms_tot*3, num_sparse))
    #b2ks_gx1_gtheta_coef = (-2./theta + (body2_features-sparse_body2_features.T)**2/theta**3)[:, :, np.newaxis]
    x_norm = body2_features-sparse_body2_features.T
    b2ks_gx1_gtheta = (-2./theta*_b2_kernel_grad_x1*b2s+x_norm**2/theta**3*b2_kernel_grad_x1.squeeze())[:, :, np.newaxis]

    for loc, b2_grad, k_grad, b2k_gx1gtheta in zip(
        body2_mapping, body2_gradients, b2_kernel_grad_x1, b2ks_gx1_gtheta
    ):
        fi, i, j = loc
        # b2_grad (1, 6) k_grad (num_sparse, 1) -> (num_sparse, 1, 6)
        curr_grad = np.repeat(k_grad[:, :, np.newaxis], 6, axis=2) * b2_grad
        print(f"curr_grad: ")
        print(curr_grad.squeeze())
        print("miaow: ")
        print(curr_grad[:, 0, 0:3].T)
        Knm_grad[i*3:i*3+3, :] += curr_grad[:, 0, 0:3].T
        print(curr_grad[:, 0, 3:6].T)
        Knm_grad[j*3:j*3+3, :] += curr_grad[:, 0, 3:6].T
        print(Knm_grad)
        # --
        Knm_gcart_gdelta[i*3:i*3+3, :] += curr_grad[:, 0, 0:3].T*2./delta
        Knm_gcart_gdelta[j*3:j*3+3, :] += curr_grad[:, 0, 3:6].T*2./delta
        # --
        #Knm_gcart_gtheta[i*3:i*3+3, :] += (curr_grad[:, 0, 0:3]*b2k_gx1gtheta).T
        #Knm_gcart_gtheta[j*3:j*3+3, :] += (curr_grad[:, 0, 3:6]*b2k_gx1gtheta).T
        #xxx = b2_grad[:, 0:3]*b2k_gx1gtheta
        #print(xxx.shape)
        #exit()
        Knm_gcart_gtheta[i*3:i*3+3, :] += (b2_grad[:, 0:3]*b2k_gx1gtheta).T
        Knm_gcart_gtheta[j*3:j*3+3, :] += (b2_grad[:, 3:6]*b2k_gx1gtheta).T
    Knm_frc = -Knm_grad

    Knm = np.vstack([Knm_ene, Knm_frc])
    print("Knm: ")
    print(Knm)
    #print(Knm.shape)
    #print(Knm_ene_gdelta)
    #print(Knm_gcart_gdelta)
    #print(Knm_gcart_gdelta.shape)
    #print(Knm_ene_gtheta)
    #print(Knm_gcart_gtheta)
    #print(Knm_gcart_gtheta.shape)
    # ---

    # --- construct Kmm
    Kmm, _, Kmm_gdelta, Kmm_gtheta  = gaussian_kernel_value_and_grad(
        sparse_body2_features, sparse_body2_features, delta=delta, theta=theta
    )
    # NOTE: add cutoff function?
    #cutoff_sparse = sparse_b2*sparse_b2.T
    #Kmm = Kmm*cutoff_sparse
    #Kmm_gdelta = Kmm_gdelta*cutoff_sparse
    #Kmm_gtheta = Kmm_gtheta*cutoff_sparse

    #print("Kmm shape: ", Kmm.shape)

    return Kmm, Knm, Kmm_gdelta, Kmm_gtheta, Knm_ene_gdelta, Knm_ene_gtheta, Knm_gcart_gdelta, Knm_gcart_gtheta

def compute_b2_marginal_likelihood(
    params, sigma2, r_cut, y_data, nframes, natoms_tot, 
    body2_features, body2_gradients, body2_mapping, sparse_body2_features
):
    delta, theta = params

    num_points = y_data.shape[0]

    Kmm, Knm, Kmm_gdelta, Kmm_gtheta, Knm_ene_gdelta, Knm_ene_gtheta, Knm_gcart_gdelta, Knm_gcart_gtheta = compute_body2_kernel_matrices(
        delta, theta, r_cut, nframes, natoms_tot, 
        body2_features, body2_gradients, body2_mapping, sparse_body2_features
    )
    Kmm_inv = np.linalg.inv(Kmm)

    #print("Knm@Kmn: ")
    #print(np.linalg.det(Knm@Knm.T))
    #print(np.linalg.det(Knm[:nframes, :]@Knm[:nframes, :].T))

    Knn = np.dot(Knm, np.dot(Kmm_inv, Knm.T)) + sigma2
    Knn_inv = np.linalg.inv(Knn)
    print("Knn: ")
    print(Knn)
    np.savetxt("./xxx.dat", Knn, fmt="%12.4f")
    print(np.linalg.det(Knn))
    #print(Knn.shape)
    exit()

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

    return -loss[0][0], [-loss_gdelta, -loss_gtheta]

def compute_b3_marginal_likelihood(
    params, sigma2, r_cut, y_data, nframes, natoms_tot, 
    b3_features, b3_gradients, b3_mapping, sparse_b3_features
):
    delta, theta = params

    num_points = y_data.shape[0]

    Kmm, Knm, Kmm_gdelta, Kmm_gtheta, Knm_ene_gdelta, Knm_ene_gtheta, Knm_gcart_gdelta, Knm_gcart_gtheta = compute_body3_kernel_matrices(
        delta, theta, r_cut, nframes, natoms_tot, 
        b3_features, b3_gradients, b3_mapping, sparse_b3_features
    )

    Kmm_inv = np.linalg.inv(Kmm)
    Knn = np.dot(Knm, np.dot(Kmm_inv, Knm.T)) + sigma2
    Knn_inv = np.linalg.inv(Knn)

    print("Knn: ")
    print(Knn)
    print(np.linalg.det(Knn))

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

    # NOTE: LOSS should be a scalar otherwise bug?
    return -loss[0][0], [-loss_gdelta, -loss_gtheta]

def compute_b2b3_marginal_likelihood(
    params, sigma2, r_cut, y_data, nframes, natoms_tot, 
    body2_features, body2_gradients, body2_mapping, sparse_b2_features, 
    body3_features, body3_gradients, body3_mapping, sparse_b3_features
):
    # -
    b2_delta, b2_theta, b3_delta, b3_theta = params

    # --
    num_points = y_data.shape[0]

    # - body2
    (
        b2_Kmm, b2_Knm, 
        b2_Kmm_gdelta, b2_Kmm_gtheta, 
        b2_Knm_ene_gdelta, b2_Knm_ene_gtheta, 
        b2_Knm_gcart_gdelta, b2_Knm_gcart_gtheta
    ) = compute_body2_kernel_matrices(
        b2_delta, b2_theta, r_cut, nframes, natoms_tot, 
        body2_features, body2_gradients, body2_mapping, sparse_b2_features
    )
    b2_Knm_gdelta = np.vstack([b2_Knm_ene_gdelta, -b2_Knm_gcart_gdelta])
    b2_Knm_gtheta = np.vstack([b2_Knm_ene_gtheta, -b2_Knm_gcart_gtheta])

    # -
    (
        b3_Kmm, b3_Knm, 
        b3_Kmm_gdelta, b3_Kmm_gtheta, 
        b3_Knm_ene_gdelta, b3_Knm_ene_gtheta, 
        b3_Knm_gcart_gdelta, b3_Knm_gcart_gtheta
    ) = compute_body3_kernel_matrices(
        b3_delta, b3_theta, r_cut, nframes, natoms_tot, 
        body3_features, body3_gradients, body3_mapping, sparse_b3_features
    )
    b3_Knm_gdelta = np.vstack([b3_Knm_ene_gdelta, -b3_Knm_gcart_gdelta])
    b3_Knm_gtheta = np.vstack([b3_Knm_ene_gtheta, -b3_Knm_gcart_gtheta])

    num_b2_sparse = sparse_b2_features.shape[0]
    num_b3_sparse = sparse_b3_features.shape[0]

    #loss, lg_b2delta, lg_b2theta, lg_b3delta, lg_b3theta = train_ene_and_frc(
    #    nframes, num_points, num_b2_sparse, num_b3_sparse, y_data, sigma2, b2_Kmm, b3_Kmm, b2_Knm, b3_Knm,
    #    b2_Kmm_gdelta, b2_Kmm_gtheta, b2_Knm_gdelta, b2_Knm_gtheta,
    #    b3_Kmm_gdelta, b3_Kmm_gtheta, b3_Knm_gdelta, b3_Knm_gtheta,
    #)
    loss, lg_b2delta, lg_b2theta, lg_b3delta, lg_b3theta = train_ene(
        nframes, num_points, num_b2_sparse, num_b3_sparse, y_data, sigma2, b2_Kmm, b3_Kmm, b2_Knm, b3_Knm,
        b2_Kmm_gdelta, b2_Kmm_gtheta, b2_Knm_gdelta, b2_Knm_gtheta,
        b3_Kmm_gdelta, b3_Kmm_gtheta, b3_Knm_gdelta, b3_Knm_gtheta,
    )

    return -loss[0][0], [-lg_b2delta, -lg_b2theta, -lg_b3delta, -lg_b3theta]

def train_ene(
    nframes, num_points, num_b2_sparse, num_b3_sparse, y_data, sigma2, b2_Kmm, b3_Kmm, b2_Knm, b3_Knm,
    b2_Kmm_gdelta, b2_Kmm_gtheta, b2_Knm_gdelta, b2_Knm_gtheta,
    b3_Kmm_gdelta, b3_Kmm_gtheta, b3_Knm_gdelta, b3_Knm_gtheta,
):
    # - train ene
    Kmm = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm[:num_b2_sparse, :num_b2_sparse] = b2_Kmm
    Kmm[num_b2_sparse:, num_b2_sparse:] = b3_Kmm

    Knm = np.zeros((nframes, num_b2_sparse+num_b3_sparse))
    Knm[:, :num_b2_sparse] = b2_Knm[:nframes, :]
    Knm[:, num_b2_sparse:] = b3_Knm[:nframes, :]

    Kmm_inv = np.linalg.inv(Kmm)
    Knn = Knm@Kmm_inv@Knm.T + sigma2[:nframes, :nframes]
    print("Knn: ")
    print(np.linalg.det(Knn))

    Knn_inv = np.linalg.inv(Knn)

    y_data_ene = y_data[:nframes, :]

    sign, logabsdet = np.linalg.slogdet(Knn)
    loss = -0.5*(np.log(sign)+logabsdet) - 0.5 * y_data_ene.T @ Knn_inv @ y_data_ene - nframes/2.*np.log(2*np.pi)

    Ky = Knn_inv @ y_data_ene

    Kmm_gdelta = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm_gdelta[:num_b2_sparse, :num_b2_sparse] = b2_Kmm_gdelta
    Knm_gdelta = np.zeros((nframes, num_b2_sparse+num_b3_sparse))
    Knm_gdelta[:, :num_b2_sparse] = b2_Knm_gdelta[:nframes, :]
    lg_b2delta = compute_loss_gradient(Ky, Knn_inv, Kmm_inv, Kmm_gdelta, Knm, Knm_gdelta)

    Kmm_gtheta = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm_gtheta[:num_b2_sparse, :num_b2_sparse] = b2_Kmm_gtheta
    Knm_gtheta = np.zeros((nframes, num_b2_sparse+num_b3_sparse))
    Knm_gtheta[:, :num_b2_sparse] = b2_Knm_gtheta[:nframes, :]
    lg_b2theta = compute_loss_gradient(Ky, Knn_inv, Kmm_inv, Kmm_gtheta, Knm, Knm_gtheta)

    Kmm_gdelta = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm_gdelta[num_b2_sparse:, num_b2_sparse:] = b3_Kmm_gdelta
    Knm_gdelta = np.zeros((nframes, num_b2_sparse+num_b3_sparse))
    Knm_gdelta[:, num_b2_sparse:] = b3_Knm_gdelta[:nframes, :]
    lg_b3delta = compute_loss_gradient(Ky, Knn_inv, Kmm_inv, Kmm_gdelta, Knm, Knm_gdelta)

    Kmm_gtheta = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm_gtheta[num_b2_sparse:, num_b2_sparse:] = b3_Kmm_gtheta
    Knm_gtheta = np.zeros((nframes, num_b2_sparse+num_b3_sparse))
    Knm_gtheta[:, num_b2_sparse:] = b3_Knm_gtheta[:nframes, :]
    lg_b3theta = compute_loss_gradient(Ky, Knn_inv, Kmm_inv, Kmm_gtheta, Knm, Knm_gtheta)

    return loss, lg_b2delta, lg_b2theta, lg_b3delta, lg_b3theta

def train_ene_and_frc(
    nframes, num_points, num_b2_sparse, num_b3_sparse, y_data, sigma2, b2_Kmm, b3_Kmm, b2_Knm, b3_Knm,
    b2_Kmm_gdelta, b2_Kmm_gtheta, b2_Knm_gdelta, b2_Knm_gtheta,
    b3_Kmm_gdelta, b3_Kmm_gtheta, b3_Knm_gdelta, b3_Knm_gtheta,
):
    """"""
    Kmm = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm[:num_b2_sparse, :num_b2_sparse] = b2_Kmm
    Kmm[num_b2_sparse:, num_b2_sparse:] = b3_Kmm

    Knm = np.zeros((num_points, num_b2_sparse+num_b3_sparse))
    Knm[:, :num_b2_sparse] = b2_Knm
    Knm[:, num_b2_sparse:] = b3_Knm

    # - combine
    Kmm_inv = np.linalg.inv(Kmm)
    print("Kmm_inv: ")
    print(np.linalg.det(Kmm_inv))

    # --- approximate Knn
    # -0.
    #print("Knm@Kmn: ")
    #print(np.linalg.det(Knm@Knm.T))
    #print(np.linalg.det(Knm[:nframes, :]@Knm[:nframes, :].T))
    #eigens, eig_vecs = np.linalg.eig(Kmm)
    #eigens = eigens[:, np.newaxis]
    #mu = np.sqrt((num_b2_sparse+num_b3_sparse)/num_points)*eigens**-1*Knm@eig_vecs
    #Knn = (num_points/(num_b2_sparse+num_b3_sparse))*eigens*mu@mu.T
    #Knn += sigma2
    #Knn_inv = np.linalg.inv(Knn)

    # -1.
    Knn = Knm@Kmm_inv@Knm.T + sigma2
    Knn_inv = np.linalg.inv(Knn)

    # -2.
    #Knn = Kmm
    #Knn_inv = Kmm_inv

    print("Knn: ")
    #print(Knn)
    print(np.linalg.det(Knn))
    #print(np.log(np.linalg.det(Knn)))
    #print(np.linalg.slogdet(Knn))
    sign, logabsdet = np.linalg.slogdet(Knn)
    #print(sign * np.exp(logabsdet))
    #exit()
    # --- approximate Knn

    # NOTE: det(A) will be zero if there are rows with zeros since
    #       some atoms may not in any three-body...
    #loss = -0.5*np.log(np.linalg.det(Knn)) - 0.5 * y_data.T @ Knn_inv @ y_data - num_points/2.*np.log(2*np.pi)
    loss = -0.5*(np.log(sign)+logabsdet) - 0.5 * y_data.T @ Knn_inv @ y_data - num_points/2.*np.log(2*np.pi)
    #loss = - 0.5 * y_data.T @ Knn_inv @ y_data - num_points/2.*np.log(2*np.pi)

    # -- combine gradients
    Ky = Knn_inv @ y_data

    Kmm_gdelta = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm_gdelta[:num_b2_sparse, :num_b2_sparse] = b2_Kmm_gdelta
    Knm_gdelta = np.zeros((num_points, num_b2_sparse+num_b3_sparse))
    Knm_gdelta[:, :num_b2_sparse] = b2_Knm_gdelta
    lg_b2delta = compute_loss_gradient(Ky, Knn_inv, Kmm_inv, Kmm_gdelta, Knm, Knm_gdelta)

    Kmm_gtheta = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm_gtheta[:num_b2_sparse, :num_b2_sparse] = b2_Kmm_gtheta
    Knm_gtheta = np.zeros((num_points, num_b2_sparse+num_b3_sparse))
    Knm_gtheta[:, :num_b2_sparse] = b2_Knm_gtheta
    lg_b2theta = compute_loss_gradient(Ky, Knn_inv, Kmm_inv, Kmm_gtheta, Knm, Knm_gtheta)

    Kmm_gdelta = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm_gdelta[num_b2_sparse:, num_b2_sparse:] = b3_Kmm_gdelta
    Knm_gdelta = np.zeros((num_points, num_b2_sparse+num_b3_sparse))
    Knm_gdelta[:, num_b2_sparse:] = b3_Knm_gdelta
    lg_b3delta = compute_loss_gradient(Ky, Knn_inv, Kmm_inv, Kmm_gdelta, Knm, Knm_gdelta)

    Kmm_gtheta = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
    Kmm_gtheta[num_b2_sparse:, num_b2_sparse:] = b3_Kmm_gtheta
    Knm_gtheta = np.zeros((num_points, num_b2_sparse+num_b3_sparse))
    Knm_gtheta[:, num_b2_sparse:] = b3_Knm_gtheta
    lg_b3theta = compute_loss_gradient(Ky, Knn_inv, Kmm_inv, Kmm_gtheta, Knm, Knm_gtheta)

    return loss, lg_b2delta, lg_b2theta, lg_b3delta, lg_b3theta

def compute_loss_gradient(Ky, Knn_inv, Kmm_inv, Kmm_gdelta, Knm, Knm_gdelta):
    """"""
    Kmm_inv_grad_wrt_delta = -Kmm_inv @ Kmm_gdelta @ Kmm_inv
    Knn_grad_wrt_delta = (
        (Knm_gdelta @ Kmm_inv + Knm @ Kmm_inv_grad_wrt_delta) @ Knm.T +
        Knm @ Kmm_inv @ Knm_gdelta.T
    )

    loss_gdelta = 0.5*np.trace((Ky @ Ky.T - Knn_inv) @ Knn_grad_wrt_delta)

    return loss_gdelta


class SparseGaussianProcessTrainer():

    def __init__(self) -> None:
        """"""
        self.r_cut = 4.0
        #self.r_cut = 6.8
        #self.r_cut = 8.0

        return
    
    def _prepare_dataset(self, ):
        """"""
        # - read dataset
        #frames = read("./dft/O2.xyz", ":1")
        #frames = read("./dft/Cu4.xyz", ":1")
        frames = read("./dft/Cu4.xyz", "1:2")
        #frames = read("./dft/Cu13.xyz", ":")
        #frames = read("./md/Cu4/cand0/traj.xyz", ":")
        #frames = read("./Cu100/cand0/traj.xyz", ":5")

        energies = [a.get_potential_energy() for a in frames]
        energies = np.array(energies)[:, np.newaxis]

        forces = np.vstack([a.get_forces() for a in frames])
        forces = forces.flatten()[:, np.newaxis]
        print(f"force shape: {forces.shape}")

        y_data = np.vstack([energies, forces])
        print(y_data)

        # - regularization
        sigma2_diag_ene = 0.05*np.sqrt(np.array([len(a) for a in frames]))
        f_sig, f_min = 0.01, 0.01
        sigma2_diag_frc = []
        for x in np.fabs(forces.flatten()):
            if x >= f_min:
                sigma2_diag_frc.append(f_sig*x)
            else:
                sigma2_diag_frc.append(f_sig*f_min)
        sigma2_diag = sigma2_diag_ene.tolist() + sigma2_diag_frc
        sigma2 = np.diag(sigma2_diag)**2

        return frames, y_data, sigma2
    
    def run(self, *args, **kwargs):
        """"""
        # - read dataset
        frames, y_data, sigma2 = self._prepare_dataset()
        nframes = len(frames)
        natoms_list = [len(a) for a in frames]
        natoms_tot = np.sum(natoms_list)

        num_points = nframes + natoms_tot*3

        # - get atomic environments
        (
            body2_mapping, body2_features, body2_gradients, body3_mapping, body3_features, body3_gradients
        ) = compute_body3_descriptor(frames, self.r_cut)
        #print("BODY3 MAPPING: ")
        #print(body3_mapping)

        #print("BODY3 GRADIENTS: ")
        #print(body3_gradients)

        # - sparsify
        sparse_body2_features = np.loadtxt("./b2_sparse.dat")[:, np.newaxis]

        print("BODY3 SHAPE: ")
        print(body3_features.shape)
        from gdpx.selector.cur import cur_selection, fps_selection
        #selected_indices = range(0, 30, 3)
        #_, selected_indices = cur_selection(body3_features, num=20, zeta=-1)
        #_, selected_indices = fps_selection(body3_features, num=10, min_distance=0.5)
        #sparse_body3_features = body3_features[selected_indices, :]
        sparse_body3_features = np.loadtxt("./backup/b3_sparse.dat")
        print(sparse_body3_features)

        # - construct matrix
        # -- parameters
        delta, theta = 0.1, 0.5

        # --- body2
        print("~~~~~ USE BODY-2 ~~~~~")
        Kmm_b2, Knm_b2, _, _, _, _, _, _ = compute_body2_kernel_matrices(
            delta, theta, self.r_cut, nframes, natoms_tot, 
            body2_features, body2_gradients, body2_mapping, sparse_body2_features
        )

        # --- body3
        print("~~~~~ USE BODY-3 ~~~~~")
        Kmm_b3, Knm_b3, _, _, _, _, _, _ = compute_body3_kernel_matrices(
            delta, theta, self.r_cut, nframes, natoms_tot, 
            body3_features, body3_gradients, body3_mapping, sparse_body3_features
        )

        # -- combine matrices
        num_b2_sparse = sparse_body2_features.shape[0]
        num_b3_sparse = sparse_body3_features.shape[0]

        Kmm = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
        Kmm[:num_b2_sparse, :num_b2_sparse] = Kmm_b2
        Kmm[num_b2_sparse:, num_b2_sparse:] = Kmm_b3

        Knm = np.zeros((num_points, num_b2_sparse+num_b3_sparse))
        Knm[:, :num_b2_sparse] = Knm_b2
        Knm[:, num_b2_sparse:] = Knm_b3

        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)

        ret = compute_b2b3_marginal_likelihood(
            [delta, theta, delta, theta],
            sigma2, self.r_cut, y_data, nframes, natoms_tot, 
            body2_features, body2_gradients, body2_mapping, sparse_body2_features, 
            body3_features, body3_gradients, body3_mapping, sparse_body3_features
        )
        print(ret)

        # ---
        res = optimize.minimize(
            compute_b2b3_marginal_likelihood, [delta, theta, delta, theta],
            args=(
                sigma2, self.r_cut, y_data, nframes, natoms_tot, 
                body2_features, body2_gradients, body2_mapping, sparse_body2_features, 
                body3_features, body3_gradients, body3_mapping, sparse_body3_features
            ), jac=True, options={"disp": True, "maxiter": 10}
        )
        print("OPT INFO: ")
        print(res)

        # ---
        params = res.x
        Kmm_b2, Knm_b2, _, _, _, _, _, _ = compute_body2_kernel_matrices(
            params[0], params[1], self.r_cut, nframes, natoms_tot, 
            body2_features, body2_gradients, body2_mapping, sparse_body2_features
        )
        Kmm_b3, Knm_b3, _, _, _, _, _, _ = compute_body3_kernel_matrices(
            params[2], params[3], self.r_cut, nframes, natoms_tot, 
            body3_features, body3_gradients, body3_mapping, sparse_body3_features
        )

        # -- combine matrices
        num_b2_sparse = sparse_body2_features.shape[0]
        num_b3_sparse = sparse_body3_features.shape[0]

        Kmm = np.zeros((num_b2_sparse+num_b3_sparse, num_b2_sparse+num_b3_sparse))
        Kmm[:num_b2_sparse, :num_b2_sparse] = Kmm_b2
        Kmm[num_b2_sparse:, num_b2_sparse:] = Kmm_b3

        Knm = np.zeros((num_points, num_b2_sparse+num_b3_sparse))
        Knm[:, :num_b2_sparse] = Knm_b2
        Knm[:, num_b2_sparse:] = Knm_b3

        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)

        return

    def run_b2(self, *args, **kwargs):
        """"""
        # - read dataset
        frames, y_data, sigma2 = self._prepare_dataset()
        nframes = len(frames)
        natoms_list = [len(a) for a in frames]
        natoms_tot = np.sum(natoms_list)

        # - get atomic environments
        (
            body2_mapping, body2_features, body2_gradients, body3_mapping, body3_features, body3_gradients
        ) = compute_body3_descriptor(frames, self.r_cut)
        print(body2_mapping)

        # - construct matrix
        # -- parameters
        delta, theta = 0.5, 0.8
        #delta, theta = 100, 1000

        # --- body2
        print("~~~~~ USE BODY-2 ~~~~~")
        sparse_body2_features = np.array([0.5, 1.0, 1.5, 2.0, 2.5])[:, np.newaxis]
        Kmm_b2, Knm_b2, _, _, _, _, _, _ = compute_body2_kernel_matrices(
            delta, theta, self.r_cut, nframes, natoms_tot, 
            body2_features, body2_gradients, body2_mapping, sparse_body2_features
        )

        # - init train
        Kmm = Kmm_b2
        Knm = Knm_b2
        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)

        ret = compute_b2_marginal_likelihood(
            [delta, theta],
            sigma2, self.r_cut, y_data, nframes, natoms_tot, 
            body2_features, body2_gradients, body2_mapping, sparse_body2_features
        )
        print([delta, theta])
        print(ret)

        res = optimize.minimize(
            compute_b2_marginal_likelihood, [delta, theta],
            args=(
                sigma2, self.r_cut, y_data, nframes, natoms_tot, 
                body2_features, body2_gradients, body2_mapping, sparse_body2_features
            ), jac=True, options={"disp": True}
        )
        print("OPT INFO: ")
        print(res)
        delta, theta = res.x
        Kmm_b2, Knm_b2, _, _, _, _, _, _ = compute_body2_kernel_matrices(
            delta, theta, self.r_cut, nframes, natoms_tot, 
            body2_features, body2_gradients, body2_mapping, sparse_body2_features
        )
        Kmm = Kmm_b2
        Knm = Knm_b2
        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)

        return

    def run_b3(self, *args, **kwargs):
        """"""
        # - read dataset
        frames, y_data, sigma2 = self._prepare_dataset()
        nframes = len(frames)
        natoms_list = [len(a) for a in frames]
        natoms_tot = np.sum(natoms_list)

        num_points = nframes + natoms_tot*3

        # - get atomic environments
        (
            body2_mapping, body2_features, body2_gradients, body3_mapping_, body3_features_, body3_gradients_
        ) = compute_body3_descriptor(frames, self.r_cut)

        # - remove duplicates?
        candidates = [0, 1]
        body3_mapping = [body3_mapping_[c] for c in candidates]
        body3_features = body3_features_[candidates, :]
        body3_gradients = body3_gradients_[candidates, :]

        print("BODY3 SHAPE: ")
        print(body3_features.shape)
        #sparse_body3_features = body3_features[:100, :]

        # - construct matrix
        # -- parameters
        delta, theta = 0.5, 0.8

        # --- body3
        print("~~~~~ USE BODY-3 ~~~~~")
        sparse_body3_features = np.array(
            [
                [1, 2, 3],
                [1.5, 2.5, 3.5],
                [2, 3, 4],
                [2.5, 3.5, 4.5],
                [4, 5, 6],
            ]
        )
        Kmm_b3, Knm_b3, _, _, _, _, _, _ = compute_body3_kernel_matrices(
            delta, theta, self.r_cut, nframes, natoms_tot, 
            body3_features, body3_gradients, body3_mapping, sparse_body3_features
        )
        # - init train
        Kmm = Kmm_b3
        Knm = Knm_b3
        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)

        ret = compute_b3_marginal_likelihood(
            [delta, theta],
            sigma2, self.r_cut, y_data, nframes, natoms_tot, 
            body3_features, body3_gradients, body3_mapping, sparse_body3_features
        ) 
        print([delta, theta])
        print(ret)

        res = optimize.minimize(
            compute_b3_marginal_likelihood, [delta, theta],
            args=(
                sigma2, self.r_cut, y_data, nframes, natoms_tot, 
                body3_features, body3_gradients, body3_mapping, sparse_body3_features
            ), jac=True, options={"disp": True, "maxiter": 100}
        )
        print("OPT INFO: ")
        print(res)
        delta, theta = res.x
        Kmm_b3, Knm_b3, _, _, _, _, _, _ = compute_body3_kernel_matrices(
            delta, theta, self.r_cut, nframes, natoms_tot, 
            body3_features, body3_gradients, body3_mapping, sparse_body3_features
        )
        Kmm = Kmm_b3
        Knm = Knm_b3

        # - train
        self._train_and_predict(nframes, sigma2, y_data, Kmm, Knm)

        return
    
    def _prev_run2(self, *args, **kwargs):
        """"""
        # - read dataset
        frames, y_data, sigma2 = self._prepare_dataset()
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
    
    def _train_and_predict(self, nframes, sigma2_matrix, y_data, Kmm, Knm):
        """"""
        jitter = 1e-5*np.eye(Kmm.shape[0])
        inverseLamb = np.linalg.inv(sigma2_matrix)

        # - direct inversion
        #weights = (
        #    np.linalg.inv(Kmm + jitter + Knm.T@inverseLamb@Knm) @ Knm.T @ inverseLamb
        #)
        #weights = np.dot(weights, y_data)
        #print("weights: ")
        #print(weights)

        # - decomposition
        wKmn = Knm.T@inverseLamb
        Qmm = Kmm + jitter + wKmn@Knm
        Vm = wKmn@y_data
        from scipy.linalg import solve
        weights = solve(Qmm, Vm, assume_a="pos")

        # - test
        print("===== TEST =====")
        #print("DFT: ")
        dft_data = y_data.flatten()
        #print("SGP: ")
        predictions = np.dot(Knm, weights)
        sgp_data  = predictions.flatten()
        print("ERR: ")
        ene_rmse = np.sqrt(np.sum((dft_data[:nframes] - sgp_data[:nframes])**2))
        ene_svar = np.sqrt(np.var((dft_data[:nframes] - sgp_data[:nframes])**2))
        print(f"ene_rmse: {ene_rmse} _ {ene_svar}")
        frc_rmse = np.sqrt(np.sum((dft_data[nframes:] - sgp_data[nframes:])**2))
        frc_svar = np.sqrt(np.var((dft_data[nframes:] - sgp_data[nframes:])**2))
        print(f"frc_rmse: {frc_rmse} _ {frc_svar}")

        return


class SGPTrainer(AbstractTrainer):

    name = "sgp"

    def __init__(
        self, config: dict, type_list: List[str] = None, train_epochs: int = 200, 
        directory=".", command="train", freeze_command="freeze", random_seed: int = None, 
        *args, **kwargs
    ) -> None:
        """"""
        super().__init__(
            config, type_list, train_epochs, 
            directory, command, freeze_command, 
            random_seed, *args, **kwargs
        )

        return

    @property
    def frozen_name(self):
        """"""
        return f"{self.name}.pb"

    def _resolve_freeze_command(self, *args, **kwargs):
        return super()._resolve_freeze_command(*args, **kwargs)

    def _resolve_train_command(self, *args, **kwargs):
        """"""
        command = self.command

        return command
    
    def train(self, dataset, init_model=None, *args, **kwargs):
        """"""
        self._print("miaow")
        sgp_trainer = SparseGaussianProcessTrainer()
        #print("!!!!! BODY-2 !!!!!") 
        #sgp_trainer._prev_run2()
        #print("!!!!! BODY-2 !!!!!") 
        #sgp_trainer.run_b2()
        #print("!!!!! BODY-3 !!!!!") 
        sgp_trainer.run_b3()
        #print("!!!!! BODY-2+3 !!!!!") 
        #sgp_trainer.run()

        return
    
    def write_input(self, dataset, *args, **kwargs):
        return super().write_input(dataset, *args, **kwargs)

    def read_convergence(self) -> bool:
        return super().read_convergence()


if __name__ == "__main__":
    ...