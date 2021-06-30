#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import os
import json 
from pathlib import Path

import numpy as np 
from scipy.sparse.linalg import LinearOperator, svds

from ase.io import read, write

from dscribe.descriptors import SOAP

from dprss.utils import (
    read_arrays, show_build_progress 
)


def descriptor_svd(at_descs, num, do_vectors='vh'):
    def mv(v):
        return np.dot(at_descs, v)
    def rmv(v):
        return np.dot(at_descs.T, v)

    A = LinearOperator(at_descs.shape, matvec=mv, rmatvec=rmv, matmat=mv)

    return svds(A, k=num, return_singular_vectors=do_vectors) # sparse SVD

def cur_selection(features, num, zeta=4, strategy='stochastic'):
    """ input should be [nframes,nfeatures]
    """
    # column vectors of descriptors
    assert features.ndim == 2
    nframes = features.shape[0]

    at_descs = features.copy().T

    # do SVD on kernel if desired
    if zeta > 0.0:
        m = np.matmul(at_descs.T, at_descs)**zeta
    else:
        m = at_descs

    print("Number of frames ", nframes)
    print("The descriptor vector shape ", at_descs.shape)
    print("The similarity matrix shape ", m.shape)
    print("Number of selected ", num)

    (u, s, vt) = descriptor_svd(m, min(max(1,int(num/2)),min(m.shape)-1))
    c_scores = np.sum(vt**2, axis=0)/vt.shape[0]

    if strategy == 'stochastic':
        selected = sorted(
            np.random.choice(range(nframes), size=num, replace=False, p=c_scores)
        )
    elif strategy == 'descent':
        selected = sorted(np.argsort(c_scores)[-num:])
    else:
        raise ValueError('Unsupport CUR selection strategy.')
    
    # check errors
    #if True:
    #    C = m[:, num]
    #    # equivalent to
    #    # Cp = np.linalg.pinv(C)
    #    # err = np.sqrt(np.sum((X - np.dot(np.dot(X, Cp), C))**2))
    #    err = np.sqrt(
    #        np.sum((m - np.dot(np.linalg.lstsq(C.T, m.T, rcond=None)[0].T, C)) ** 2)
    #    )
    #    print("Reconstruction RMSE={:.3e}".format(err))

    return c_scores, selected

def calc_feature(frames, soap_parameters, njobs=1, saved_npy='features.npy'):
    """ Calculate feature vector for each configuration 
    """
    soap = SOAP(
        **soap_parameters
    )

    print(soap.get_number_of_features())

    # TODO: must use outer average? more flexible features 
    soap_instances = soap.create(frames, n_jobs=njobs)
    #for cur_soap, atoms in zip(soap_instances, frames):
    #    atoms.info['feature_vector'] = cur_soap

    # save calculated features 
    np.save(saved_npy, soap_instances)

    print('number of soap instances', len(soap_instances))

    return soap_instances

def select_structures(
        random_structures, 
        param_json, num, njobs=1, feature_exist=True, 
        output='selected_structures.xyz'
    ):
    """ calculate configurational feature and select configurations 
    """
    cwd = Path.cwd() 
    # read descriptor hyperparameter 
    with open(param_json) as fopen:
        params = json.load(fopen)

    soap_parameters = params['soap']

    # read structures and calculate features 
    frames = read(random_structures, ':')
    if feature_exist:
        features = np.load(cwd / 'features.npy')
    else:
        print('start calculating features...')
        features = calc_feature(frames, soap_parameters, njobs)
        print('finished calculating features...')
        exit()

    # cur decomposition 
    zeta, strategy = params['selection']['zeta'], params['selection']['strategy']

    cur_scores, selected = cur_selection(features, num, zeta, strategy)
    content = '# idx cur sel\n'
    for idx, cur_score in enumerate(cur_scores):
        stat = 'F'
        if idx in selected:
            stat = 'T'
        content += '{:>12d}  {:>12.8f}  {:>2s}\n'.format(idx, cur_score, stat) 
    with open(cwd / 'cur_scores.txt', 'w') as writer:
        writer.write(content)

    #selected_frames = []
    #for idx in selected:
    #    selected_frames.append(frames[int(idx)])
    #write('selected_structures.xyz', selected_frames)
    print("Writing structure file... ")
    for idx, sidx in enumerate(selected):
        write(cwd / (cwd.name+'-sel.xyz'), frames[int(sidx)], append=True)
        show_build_progress(num, idx)
    print('')

    return 


if __name__ == '__main__':
    #frames = read('./blz_frames.xyz', ':')
    soap_parameters = {
        "species" : ["Zn", "Cr", "O"], 
        "rcut" : 6.0, 
        "nmax" : 12, 
        "lmax" : 6, 
        "sigma" : 0.5, 
        "average" : "inner", # In general, the inner averaging will preserve the configurational information better but you can experiment with both versions.
        "periodic" : True
    } 

    soap = SOAP(**soap_parameters)


    calc_feature(traj_frames, soap_parameters, njobs=16, saved_npy='features.npy')

