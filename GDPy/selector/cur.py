#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
from scipy.spatial.distance import cdist

"""Methods for selection of vector-based descriptors.
"""

# - CUR decomposition
def descriptor_svd(at_descs, num: int, do_vectors="vh"):
    """Perfrom a sparse SVD."""
    def mv(v):
        return np.dot(at_descs, v)
    def rmv(v):
        return np.dot(at_descs.T, v)

    A = LinearOperator(at_descs.shape, matvec=mv, rmatvec=rmv, matmat=mv)

    return svds(A, k=num, return_singular_vectors=do_vectors) # sparse SVD

def cur_selection(features, num: int, zeta: float=2, strategy: str="descent", rng=np.random):
    """Performa a CUR selection.

    Args:
        features: A (nframes,nfeatures) shaped arrray.
        num: Selected number.
        zeta: Exponential coefficient.
        strategy: Selection strategy, either stochastic or descent.
        rng: Random generator.

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

    (u, s, vt) = descriptor_svd(m, min(max(1,int(num/2)),min(m.shape)-1))
    c_scores = np.sum(vt**2, axis=0) / vt.shape[0]

    if strategy == "stochastic":
        selected = sorted(
            rng.choice(range(nframes), size=num, replace=False, p=c_scores)
        )
    elif strategy == "descent":
        selected = sorted(np.argsort(c_scores)[-num:])
    else:
        raise ValueError('Unsupport CUR selection strategy.')

    return c_scores, selected

# - FPS (Farthest Point Sampling)
def fps_selection(features, num: int, min_distance=0.1, metric="euclidean", metric_params={}, rng=np.random):
    """Farthest point sampling on vector-based features.

    Reference:
        https://github.com/bigd4/PyNEP.git

    """
    npoints = features.shape[0]

    selected_indices, selected_features = [], []

    # - get starting point
    start_index = rng.choice(npoints)
    selected_indices.append(start_index)
    selected_features.append(features[start_index])

    distances = np.min(
        cdist(features, selected_features, metric=metric, **metric_params), axis=1
    ) # shape (npoints,nselected) -> (npoints,)

    while np.max(distances) > min_distance or len(selected_indices) < num:
        i = np.argmax(distances)
        selected_indices.append(i)
        if len(selected_indices) >= num:
            break
        distances = np.minimum(
            distances, cdist([features[i]], features, metric=metric, **metric_params)[0]
        ) # shape (npoints,)
    
    scores = distances

    return scores, selected_indices

if __name__ == "__main__":
    pass