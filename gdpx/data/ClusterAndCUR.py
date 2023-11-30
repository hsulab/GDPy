#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import argparse
from itertools import groupby

import numpy as np

"""
Following the below protocol, 
we analyse and reduce the size of dataset.

1. Use SOAP to represent each structure.
2. Use HDBSCAN to cluster structures.
3. Use histogram and CUR to select most representative structures.
"""

from sklearn.datasets import make_blobs
# import pandas as pd

import hdbscan

from ase.io import read, write

from dscribe.descriptors import SOAP

from gdpx.utils.data.cur import cur_selection

def frames2features(frames, feature_computer, njobs=1, saved_file=None):
    """ use dscribe to calculate structure representation
    """
    print("feature dimension: ", feature_computer.get_number_of_features())

    # computation
    features = feature_computer.create(frames, n_jobs=njobs)

    # save calculated features 
    if saved_file is not None:
        np.save(saved_file, features)

    assert len(frames) == features.shape[0], "number of frames != number of features"

    return features


def simple_test():
    blobs, labels = make_blobs(n_samples=2000, n_features=10)
    clusterer = hdbscan.HDBSCAN()

    """
    HDBSCAN(
        algorithm='best', alpha=1.0, approx_min_span_tree=True,
        gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
        metric='euclidean', # different distance metrics
        min_cluster_size=5, # minimum number of samples in each cluster
        min_samples=None, # number of samples to form dense area, default is min_cluster_size
        p=None
    )
    """

    clusterer.fit(blobs)
    print(clusterer.labels_)

    # clusterer.probabilities_ # each sample has probs of each cluster

def cluster_features(features, frames):
    """"""
    print("===== Start Clustering =====")
    clusterer = hdbscan.HDBSCAN(
        #min_cluster_size = 32
        min_cluster_size = 8 # avoid too many small clusters
    )

    clusterer.fit(features)
    #print(clusterer.labels_)
    nclusters = len(set(clusterer.labels_)) - 1 # -1 means noises
    print("number of clusters: ", nclusters)
    #print(clusterer.probabilities_) # each sample has probs of each cluster

    #frame_clusters = []
    #for i in range(nclusters+1):
    #    frame_clusters.append([])
    #print(frame_clusters)

    index_clusters = []
    for i in range(nclusters+1):
        index_clusters.append([])
    #print(index_clusters)

    for i, idx in zip(clusterer.labels_, range(len(frames))):
        # print(i)
        index_clusters[i].append(idx)

    return index_clusters

def cluster_and_cur(frames, feature_cached=True):
    # read frames
    #frames = read("/users/40247882/scratch2/analyse-data/O24Pt24-test.xyz", ":")

    # calculate features
    with open("/users/40247882/scratch2/analyse-data/soap_param.json", 'r') as fopen:
        input_dict = json.load(fopen)

    soap = SOAP(**input_dict['soap'])
    if feature_cached:
        features = np.load("fea.npy")
        print("features are loaded from file")
    else:
        start_time = time.time()
        features = frames2features(frames, soap, njobs=4, saved_file="fea.npy")
        end_time = time.time()
        print("feature calculation time: ", end_time - start_time)

    # clustering
    start_time = time.time()
    index_clusters = cluster_features(features, frames)
    end_time = time.time()
    print("clustering time: ", end_time - start_time)
    #print(index_clusters)

    #for i in range(-1, nclusters):
    #    print("#frames %d in cluster %d" %(len(frame_clusters[i]), i))
    #    write("xxx-%d.xyz" %i, frame_clusters[i])

    for icluster, indices in enumerate(index_clusters):
        print("===== Cluster %d =====" %icluster)
        cur_frames = []
        for x in indices:
            atoms = frames[x]
            atoms.info["index"] = x
            cur_frames.append(atoms)
        write("group-"+str(icluster)+".xyz", cur_frames)
        print("Current #frames: ", len(indices))
    
    return


def reduce_group(frames, features, cluster_indices):
    """ reduce dataset by CUR decomposition
    """
    index_clusters = []
    for i in cluster_indices:
        cur_frames = read("./groups/group-"+str(i)+".xyz", ":")
        indices = [atoms.info["index"] for atoms in cur_frames]
        index_clusters.append(indices)
    # CUR decomposition for dataset reduction
    min_size = 500
    ratio = 0.5 # dataset reduction ratio
    for icluster, indices in zip(cluster_indices, index_clusters):
        print("cluster group ", icluster)
        cname = "group-"+str(icluster)+"_cured.xyz"
        if len(indices) > min_size:
            cur_features = features[indices,:]
            cur_nframes = cur_features.shape[0]
            print("Current #frames", cur_nframes)
            nselected = int(np.ceil(cur_nframes*ratio))
            print("Current #selected", nselected)
            cur_scores, cur_selected = cur_selection(cur_features, nselected, zeta=-1, strategy="descent")
            selected_indices = [indices[x] for x in cur_selected]
            # TODO: add index info to selected frames
            cur_frames = [frames[x] for x in selected_indices]
            write(cname, cur_frames)
        else:
            print("Clustersize is too small only with %d" %len(indices))
            continue
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--frames", nargs="*",
        required=True,
        help="input frames"
    )
    parser.add_argument(
        "--fea", action="store_true",
        help="if features are calculated"
    )

    parser.add_argument(
        "--cur", action="store_true",
        help="cur decomposition"
    )
    parser.add_argument(
        "-ci", "--cluster_indices", nargs="*",
        help="cur decomposition"
    )

    args = parser.parse_args()

    frames = []
    for f in args.frames:
        frames.extend(
            read(f, ":")
        )
    print("#frames: ", len(frames))

    if not args.cur:
        # clustering
        cluster_and_cur(frames, args.fea)
    else:
        features = np.load('fea.npy')
        reduce_group(frames, features, args.cluster_indices)