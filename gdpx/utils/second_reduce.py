#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from os import system
import pathlib
import argparse
from sys import prefix
from typing import overload
from ase.io.formats import F

import numpy as np

from ase.io import read, write
from ase.constraints import FixAtoms

from gdpx.selector.structure_selection import calc_feature, cur_selection, select_structures

"""
data route
Pt32 and Pt31 ->
= (2x2) surface = 
Pt111 -> FCC and HCP
= (3x3) surface = 
if still good ->
various oxides
"""

FULL_DATASET_PATH = "/users/40247882/scratch2/PtOx-dataset/"
DESC_JSON_PATH = "/users/40247882/scratch2/oxides/input-jsons/soap_param.json"

def read_frames(system_name, pattern):
    # read xyz dataset
    dataset_path = pathlib.Path(FULL_DATASET_PATH)
    # system_name = "O0Pt32"
    system_path = dataset_path / system_name # 2126, 1920, muts be times of both 32 and 10 = 160
    print(str(system_path))
    # TODO: should sort dirs to make frames consistent

    total_frames = []
    for p in system_path.glob(pattern):
        print(p)
        frames = read(p, ":")
        print("number of frames: ", len(frames))
        total_frames.extend(frames)
    print("Total number: ", len(total_frames))
    
    return total_frames

def calc_desc(frames):
    # calculate descriptor to select minimum dataset
    with open(DESC_JSON_PATH, "r") as fopen:
        desc_dict = json.load(fopen)
    desc_dict = desc_dict["soap"]

    # TODO: inputs
    njobs = 4

    cwd = pathlib.Path.cwd()
    features_path = cwd / "features.npy"
    if features_path.exists():
        print("use precalculated features...")
        features = np.load(features_path)
        assert features.shape[0] == len(frames)
    else:
        print('start calculating features...')
        features = calc_feature(frames, desc_dict, njobs, features_path)
        print('finished calculating features...')
    
    return features

def select_structures(
    system_name, features, num, zeta=-1, strategy="descent", index_map=None, prefix="",
    manually_selected = None
):
    """ 
    """
    # cur decomposition 
    cur_scores, selected = cur_selection(features, num, zeta, strategy)
    content = '# idx cur sel\n'
    for idx, cur_score in enumerate(cur_scores):
        stat = 'F'
        if idx in selected:
            stat = 'T'
        if index_map is not None:
            idx = index_map[idx]
        content += '{:>12d}  {:>12.8f}  {:>2s}\n'.format(idx, cur_score, stat) 
    with open(cwd / (prefix+"cur_scores.txt"), 'w') as writer:
        writer.write(content)

    # map selected indices
    if index_map is not None:
        selected = [index_map[s] for s in selected]
    if manually_selected is not None:
        selected.extend(manually_selected)
    np.save(cwd / (prefix+"indices.npy"), selected)

    selected_frames = []
    for idx, sidx in enumerate(selected):
        selected_frames.append(frames[int(sidx)])

    return selected_frames

def create_training():
    # train model ensemble and run MD calculation to test properties
    return

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--name", 
    help = "system name"
)
parser.add_argument(
    "-p", "--pattern", default = "*.xyz",
    help = "xyz search pattern"
)
parser.add_argument(
    "-num", "--number", 
    default = -1, type=int,
    help = "number of selection"
)
parser.add_argument(
    "-c", "--count", required = True,
    type = int,
    help = "number of selection"
)
parser.add_argument(
    "--more", action="store_true",
    help = "select more structures from the rest"
)

args = parser.parse_args()

cwd = pathlib.Path.cwd()
print("read frames from system directory")
frames = read_frames(args.name, args.pattern)

prefix = "r" + str(args.count) + "-"
if args.count == 0:
    if args.number > 0:
        converged_frames = []
        converged_indices = []
        # add converged structures
        for idx, atoms in enumerate(frames):
            # add constraint to neglect forces of frozen atoms
            cons = FixAtoms(indices=[a.index for a in atoms if a.position[2] < 2.5])
            atoms.set_constraint(cons)
            max_force = np.max(np.fabs(atoms.get_forces()))
            if max_force < 0.05:
                converged_frames.append(atoms)
                converged_indices.append(idx)

        # select frames
        nconverged = len(converged_frames)
        print("Number of converged: ", nconverged)
        features = calc_desc(frames)
        selected_frames = select_structures(
            args.name, features, args.number-nconverged, prefix=prefix,
            manually_selected = converged_indices
        ) # include manually selected structures
        # selected_frames.extend(converged_frames)
        # TODO: sometims the converged one will be selected... so duplicate...

        print("Writing structure file... ")
        write(cwd / (prefix+args.name+'-sel.xyz'), selected_frames)
        print("")
else:
    merged_indices = []
    for i in range(args.count):
        previous_indices_path = pathlib.Path("r" + str(i) + "-indices.npy")
        indices = np.load(previous_indices_path).tolist()
        merged_indices.extend(indices)
    print(
        "Number of unique frames: ",
        len(set(merged_indices))
    )

    if previous_indices_path.exists() and args.more:
        features_path = cwd / "features.npy"
        features = np.load(features_path)

        assert len(frames) == features.shape[0]

        index_map = []
        rest_features = []
        rest_frames = []
        for idx, atoms in enumerate(frames):
            if idx not in merged_indices:
                rest_frames.append(atoms)
                rest_features.append(features[idx])
                index_map.append(idx)
        if len(rest_frames) < args.number:
            raise ValueError("Not enough structures left...")
        rest_features = np.array(rest_features)
        selected_frames = select_structures(args.name, rest_features, args.number, index_map=index_map, prefix=prefix)

        print("Writing structure file... ")
        write(cwd / (prefix+args.name+'-sel.xyz'), selected_frames)
        print("")


if __name__ == "__main__":
    pass