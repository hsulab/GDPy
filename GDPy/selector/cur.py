#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds


def descriptor_svd(at_descs, num, do_vectors="vh"):
    """ sparse SVD
    """
    def mv(v):
        return np.dot(at_descs, v)
    def rmv(v):
        return np.dot(at_descs.T, v)

    A = LinearOperator(at_descs.shape, matvec=mv, rmatvec=rmv, matmat=mv)

    return svds(A, k=num, return_singular_vectors=do_vectors) # sparse SVD

def cur_selection(features, num, zeta=2, strategy="descent", rng=np.random):
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

if __name__ == "__main__":
    pass