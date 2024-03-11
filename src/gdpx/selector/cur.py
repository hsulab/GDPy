#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
from typing import List

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
from scipy.spatial.distance import cdist

"""Methods for selection of vector-based descriptors.
"""


# -
def stat_str2val(stat: str, values: List[float]) -> float:
    """Convert a statistic string to a specific value.

    Args:
        stat: Statistics name.
        values: A List of scalar values.

    Return:
        The statistics value.

    """
    if isinstance(stat, str):
        if stat == "min":
            v = np.min(values)
        elif stat == "max":
            v = np.max(values)
        elif stat == "mean" or stat == "avg":  # Compatibilty.
            v = np.mean(values)
        elif stat == "svar":
            v = np.sqrt(np.var(values - np.average(values)))
        elif stat == "median":
            v = np.median(values)
        elif stat.startswith("percentile"):
            q = int(stat.strip().split("_")[1])  # should be within 0 and 100
            v = np.percentile(values, q)
        else:
            raise RuntimeError(f"Unknown statistics {stat}.")
    else:
        if stat == -np.inf:
            v = np.min(values)
        elif stat == np.inf:
            v = np.max(values)
        else:  # assume it is a regular number or a numpy scalar
            v = stat

    return v


# - CUR decomposition
def descriptor_svd(at_descs, num: int, do_vectors="vh"):
    """Perfrom a sparse SVD."""

    def mv(v):
        return np.dot(at_descs, v)

    def rmv(v):
        return np.dot(at_descs.T, v)

    A = LinearOperator(at_descs.shape, matvec=mv, rmatvec=rmv, matmat=mv)

    return svds(A, k=num, return_singular_vectors=do_vectors)  # sparse SVD


def cur_selection(
    features, num: int, zeta: float = 2, strategy: str = "descent", rng=np.random
):
    """Performa a CUR selection.

    Args:
        features: A (nframes,nfeatures) shaped arrray.
        num: Selected number.
        zeta: Exponential coefficient.
        strategy: Selection strategy, either stochastic or descent.
        rng: Random generator.

    References:
        [1] Bernstein, N.; Csányi, G.; Deringer, V. L.
            De Novo Exploration and Self-Guided Learning of Potential-Energy Surfaces.
            npj Comput. Mater. 2019, 5, 99.
        [2] Mahoney, M. W.; Drineas, P.
            CUR Matrix Decompositions for Improved Data Analysis.
            Proc. Natl. Acad. Sci. USA 2009, 106, 697–702.

    """
    # column vectors of descriptors
    assert features.ndim == 2
    nframes = features.shape[0]

    at_descs = features.copy().T

    # do SVD on kernel if desired
    if zeta > 0.0:
        m = np.matmul(at_descs.T, at_descs) ** zeta
    else:
        m = at_descs

    (u, s, vt) = descriptor_svd(m, min(max(1, int(num / 2)), min(m.shape) - 1))
    c_scores = np.sum(vt**2, axis=0) / vt.shape[0]

    if strategy == "stochastic":
        selected = sorted(
            rng.choice(range(nframes), size=num, replace=False, p=c_scores)
        )
    elif strategy == "descent":
        selected = sorted(np.argsort(c_scores)[-num:])
    else:
        raise ValueError("Unsupport CUR selection strategy.")

    return c_scores, selected


# - FPS (Farthest Point Sampling)
def fps_selection(
    features,
    num: int,
    min_distance=0.1,
    metric="euclidean",
    metric_params={},
    rng=np.random,
):
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
    )  # shape (npoints,nselected) -> (npoints,)

    while np.max(distances) > min_distance or len(selected_indices) < num:
        i = np.argmax(distances)
        selected_indices.append(i)
        if len(selected_indices) >= num:
            break
        distances = np.minimum(
            distances, cdist([features[i]], features, metric=metric, **metric_params)[0]
        )  # shape (npoints,)

    scores = distances

    return scores, selected_indices


# - boltz (Boltzmann Selection)
def boltz_selection(
    boltz: float,
    props: List[float],
    input_indices: List[int],
    num_minima: int,
    rng=np.random,
):
    """Selected indices based on Boltzmann distribution.

    References:
        [1] Bernstein, N.; Csányi, G.; Deringer, V. L.
            De Novo Exploration and Self-Guided Learning of Potential-Energy Surfaces.
            npj Comput. Mater. 2019, 5, 99.

    """
    # compute desired probabilities for flattened histogram
    hist, bin_edges = np.histogram(props, bins=10)  # hits, bin_edges
    min_prop = np.min(props)

    # - multiply bin number
    config_prob = []
    for H in props:
        bin_i = np.searchsorted(bin_edges[1:], H)  # ret index of the bin
        if hist[bin_i] > 0.0:
            p = 1.0 / hist[bin_i]
        else:
            p = 0.0
        if boltz > 0.0:
            p *= np.exp(-(H - min_prop) / boltz)  # TODO: custom expression?
        config_prob.append(p)

    assert len(config_prob) == len(props)
    # uniform_probs = np.array(config_prob) / np.sum(config_prob)

    # - select
    props = copy.deepcopy(props)
    input_indices = copy.deepcopy(input_indices)

    scores, selected_indices = [], []
    for i in range(num_minima):
        # -- random
        # TODO: rewrite by mask
        config_prob = np.array(config_prob)
        config_prob /= np.sum(config_prob)
        cumul_prob = np.cumsum(config_prob)
        rv = rng.uniform()
        config_i = np.searchsorted(cumul_prob, rv)
        # print(converged_trajectories[config_i][0])
        selected_indices.append(input_indices[config_i])

        # -- remove from config_prob by converting to list
        scores.append(config_prob[config_i])
        config_prob = list(config_prob)
        del config_prob[config_i]

        # remove from other lists
        del props[config_i]
        del input_indices[config_i]

    # NOTE: scores are current probabilities when selected

    return scores, selected_indices


# - hist (Histogram-Based Selection)
def hist_selection(
    nbins: int,
    pmin: float,
    pmax: float,
    props: List[float],
    input_indices: List[int],
    num_minima: int,
    rng=np.random,
):
    props = np.array(props)
    scores = None
    selected_indices = None

    if pmin == -np.inf:
        pmin = props.min()
    if pmax == np.inf:
        pmax = props.max()
    # print("hist: ", pmin, pmax)

    bin_edges = np.linspace(pmin, pmax, nbins, endpoint=False).tolist()
    bin_edges.append(pmax)
    # print(len(bin_edges), bin_edges)

    bin_indices = np.digitize(props, bin_edges, right=False)
    # print("bin_indices: ", bin_indices)

    groups = [[] for _ in range(nbins)]
    for i, i_bin in enumerate(bin_indices):
        # dump prop not in pmin and pmax
        if 0 < i_bin <= nbins:
            groups[i_bin - 1].append(i)
    hist_by_digit = np.array([len(x) for x in groups])
    # print(hist_by_digit)

    # - select bins
    selected_groups = [[] for i in range(nbins)]

    cur_groups_ = copy.deepcopy(groups)
    for i in range(num_minima):
        # -- select bin
        cur_hists_ = np.array([len(x) for x in cur_groups_])
        curr_npoints = np.sum(cur_hists_)
        if curr_npoints > 0:
            # print("hist: ", cur_hists_)
            cur_probs_ = cur_hists_ / np.sum(cur_hists_)
            s_bin = rng.choice(nbins, 1, p=cur_probs_, replace=False)[0]
            # -- select index in the bin
            s_ind = rng.choice(cur_hists_[s_bin], 1, replace=False)[0]
            selected_groups[s_bin].append(cur_groups_[s_bin][s_ind])
            del cur_groups_[s_bin][s_ind]
        else:
            # Not enough data points for num_minima
            break
    # print([len(x) for x in selected_groups])

    # - select points
    selected_indices = []
    for x in selected_groups:
        selected_indices.extend(x)
    scores = [props[i] for i in selected_indices]

    return scores, selected_indices


if __name__ == "__main__":
    ...
