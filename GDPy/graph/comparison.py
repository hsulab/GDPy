#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pickle
from typing import List
from pathlib import Path

import networkx as nx

from joblib import Parallel, delayed

"""Some methods to compare graph.
"""

#: Handles isomorphism for bonds.
bond_match = nx.algorithms.isomorphism.categorical_edge_match("bond", "")

#: Handles isomorphism for atoms with regards to perodic boundary conditions.
ads_match = nx.algorithms.isomorphism.categorical_node_match(["index", "ads"], [-1, False]) 

# - serial algorithms
def get_unique_environments_based_on_bonds(chem_envs: List[nx.Graph]) -> List[int]:
    """Find unique environments based on graph edges.

    This method compares one graph with another.

    Args:
        chem_envs: List of graph representation of an atom.

    Returns:
        Indices of unique graphs.
    
    """

    nsites = len(chem_envs)

    unique_indices = [0]
    for i in range(1,nsites):
        #print("u: ", unique_indices)
        for j in unique_indices:
            env_i, env_j = chem_envs[i], chem_envs[j]
            #print(i, env_i, j, env_j)
            #if nx.algorithms.isomorphism.is_isomorphic(env_i, env_j, edge_match=bond_match, node_match=ads_match):
            if nx.algorithms.isomorphism.is_isomorphic(env_i, env_j, edge_match=bond_match):
                #print(f"{i} == {j}")
                break
        else:
            unique_indices.append(i)

    return unique_indices

def get_unique_environments_based_on_nodes_and_edges(chem_envs):
    """Unique adsorbates. Get unique adsorbate chemical environment.
    
    Removes duplicate adsorbates which occur when PBCs detect the same 
    adsorbate in two places. Each adsorbate graph has its atomic index checked 
    to detect when PBC has created duplicates.

    Args:
        chem_env_groups (list[networkx.Graph]): Adsorbate/environment graphs

    Returns:
        list[networkx.Graph]: The unique adsorbate graphs

    """
    # Keep track of known unique adsorbate graphs
    unique = []
    for env in chem_envs:
        for unique_env in unique:
            if nx.algorithms.isomorphism.is_isomorphic(
                env, unique_env, edge_match=bond_match, node_match=ads_match
            ):
                break
        else: # Was unique
            unique.append(env)

    return unique

def compare_chem_envs(chem_envs1, chem_envs2):
    """Compares two sets of chemical environments to see if they are the same
    in chemical identity.  Useful for detecting if two sets of adsorbates are
    in the same configurations.

    Args:
        chem_envs1 (list[networkx.Graph]): A list of chemical environments, 
                                           including duplicate adsorbates
        chem_envs2 (list[networkx.Graph]): A list of chemical environments, 
                                           including duplicate adsorbates

    Returns:
        bool: Is there a matching graph (site / adsorbate) for each graph?

    """
    # Do they have the same number of adsorbates?
    if len(chem_envs1) != len(chem_envs2):
        return False

    envs_copy = chem_envs2[:] # Make copy of list

    # Check if chem_envs1 matches chem_envs2 by removing from envs_copy
    for env1 in chem_envs1: 
        for env2 in envs_copy:
            if nx.algorithms.isomorphism.is_isomorphic(env1, env2, edge_match=bond_match):
                # Remove this from envs_copy and move onto next env in chem_envs1
                envs_copy.remove(env2)
                break

    # Everything should have been removed from envs_copy if everything had a match
    if len(envs_copy) > 0:
        return False

    return True

def unique_chem_envs(chem_envs_groups, metadata=None, verbose=False):
    """Given a list of chemical environments, find the unique
    environments and keep track of metadata if required.

    This function exists largely to help with unique site detection
    but its performance will scale badly with extremely large numbers
    of chemical environments to check.  This can be split into parallel
    jobs.

    Args:
        chem_env_groups (list[list[networkx.Graph]]): 
            Chemical environments to compare against each other
        metadata (list[object]): 
            Corresponding metadata to keep with each chemical environment

    Returns:
        list[list[list[networkx.Graph]]]: A list of unique chemical environments 
                                          with their duplicates
        list[list[object]]: A matching list of metadata
    """
    # print("*** org algo for chem_env cmp***")
    # Error checking, this should never really happen
    #print("chem_envs: ", chem_envs_groups)
    if len(chem_envs_groups) == 0:
        return [[],[]]

    # We have metadata to manage
    if metadata is not None:
        if len(chem_envs_groups) != len(metadata):
            raise ValueError("Metadata must be the same length as\
                              the number of chem_envs_groups")
    
    # No metadata to keep track of
    if metadata is None:
        metadata = [None] * len(chem_envs_groups)

    # Keep track of known unique environments
    unique = []

    for index, env in enumerate(chem_envs_groups):
        #st = time.time()
        for index2, (unique_indices, unique_env) in enumerate(unique):
            if verbose:
                print("Checking for uniqueness {:05d}/{:05d} {:05d}/{:05d}".format(index+1, len(chem_envs_groups), index2, len(unique)), end='\r')
            if compare_chem_envs(env, unique_env):
                unique_indices.append(index)
                break
        else: # Was unique
            if verbose:
                print("")
            unique.append(([index], env))
        #et = time.time()
        #print("used time ", et-st, " for ", index, " for ", len(unique))
    
    # Zip trick to split into two lists to return
    # unique_envs, unique_groups
    # unique_groups = [[index,metadata],[]]
    return zip(*[(env, [metadata[index] for index in indices]) for (indices, env) in unique])

# - an algorithm for parallel comparison of graph groups
def new_unique_chem_envs(extra_info, chem_envs_groups, env_indices, verbose=False):
    """"""
    if verbose:
        print("*** new org algo for chem_env cmp***")
    # Error checking, this should never really happen
    #print("chem_envs: ", chem_envs_groups)
    if len(chem_envs_groups) == 0:
        return [[],[]]

    # Keep track of known unique environments
    unique = []

    for index in env_indices:
        st = time.time()
        env = chem_envs_groups[index]

        # NOTE: use a flag since there may continue all and not break
        found_similar = False
        for ug_idx, (unique_indices, ur_idx) in enumerate(unique):
            # check if already compared
            if ur_idx in extra_info[index]["others"]:
                continue
            # add reclusive
            extra_info[index]["others"].append(ur_idx)
            extra_info[ur_idx]["others"].append(index)
            # actual compare
            if compare_chem_envs(env, chem_envs_groups[ur_idx]):
                found_similar = True
                unique_indices.append(index)
                extra_info[ur_idx]["similar"].append(index)
                #print(ur_idx, extra_info[ur_idx])
                if len(extra_info[index]["similar"]) > 0:
                    extra_info[ur_idx]["similar"].extend(extra_info[index]["similar"])
                    extra_info[index]["similar"] = []
                break

        if not found_similar: # Was unique
            unique.append(([index], index))

        et = time.time()
        if verbose:
            print("used time ", et-st, " for ", index, " for ", len(unique))
    
    return unique

def get_indices(env_indices, n_jobs):
    nframes = len(env_indices)
    size = nframes // n_jobs

    st_indices = [x*size for x in range(n_jobs)]
    end_indices = [x+size for x in st_indices]
    end_indices[-1] = nframes

    env_indices_splits = []
    for s, e in zip(st_indices, end_indices):
        env_indices_splits.append([env_indices[x] for x in range(s,e)])

    return env_indices_splits

def get_groups(extra_info, env_indices, nsplits):
    """"""
    nframes = len(env_indices)
    size = nframes // nsplits
    st_indices = [x*size for x in range(nsplits)]
    end_indices = [x+size for x in st_indices]
    end_indices[-1] = nframes

    temp_env_indices = env_indices.copy()
    env_indices_splits = []

    nvalid = 0
    while nvalid != nsplits-1:
        if len(temp_env_indices) < 1:
            break
        s, e = st_indices[nvalid], end_indices[nvalid]
        size = e - s
        #print("size: ", size)
        #print(len(temp_env_indices), temp_env_indices)
        first = temp_env_indices[0]
        del temp_env_indices[0]
        nrest = len(temp_env_indices)
        #mates = [x for x in temp_env_indices if x not in extra_info[first]["others"]]
        mates = []
        count = 0
        while len(mates) < size-1:
            if count >= nrest:
                break
            if temp_env_indices[count] not in extra_info[first]["others"]:
                mates.append(temp_env_indices[count])
            count += 1
        if len(mates) < 1:
            continue
        env_indices_splits.append([first]+mates)
        temp_env_indices = list(set(temp_env_indices).difference(mates))
        nvalid += 1

    # last group
    if len(temp_env_indices) > 0:
        env_indices_splits.append(temp_env_indices)

    return env_indices_splits

def merge_results(extra_info, chem_envs_groups, metadata):
    duplicates = []
    for i, info in enumerate(extra_info):
        if len(info["similar"]) > 0:
            duplicates.extend(info["similar"])

    unique_indices = []
    unique_envs, unique_groups = [], []
    for i, info in enumerate(extra_info):
        if i in duplicates:
            continue
        unique_indices.append(i)
        unique_envs.append(chem_envs_groups[i])
        u_indices = [i]
        if len(info["similar"]) > 0:
            u_indices += info["similar"]
        u_indices = sorted(u_indices)
        unique_groups.append([metadata[u] for u in u_indices])
    # TODO: resort

    return unique_indices, unique_envs, unique_groups


def paragroup_unique_chem_envs(chem_envs_groups, metadata=None, directory=Path.cwd(), n_jobs=1):
    """"""
    # - switch to plain version if only one job is used
    if n_jobs == 1:
        unique_envs, unique_groups = unique_chem_envs(chem_envs_groups, metadata)
        return unique_envs, unique_groups

    # - run parallel version, not always correct but can reduce most duplicates
    nframes = len(chem_envs_groups)

    directory = Path(directory)
    saved_name = "extra_info.pkl"
    saved_info = directory / "extra_info.pkl"
    if saved_info.exists():
        with open(saved_info, "rb") as fopen:
            extra_info = pickle.load(fopen)
        env_indices, unique_envs, unique_groups = merge_results(extra_info, chem_envs_groups, metadata)
        print("load cached comparasion info...")
    else:
        extra_info = [
            {"similar": [], "others": []} for i in range(nframes)
        ]
        env_indices = list(range(nframes))

    # determine atomic task size
    # NOTE: MAYBE A BUG? 400/64 may miss some comparasions
    # that is for too small atomic task
    #nsplits = 64
    nsplits = n_jobs*(nframes // 16 // n_jobs)
    if nsplits == 0:
        nsplits += 1
    print("nsplits: ", nsplits)
    
    with Parallel(
        n_jobs=n_jobs, 
        prefer="threads",
        #require="sharedmem"
    ) as para:
        st = time.time()
        # split data

        for icount in range(500):
            print(f"--- {icount} ---")
            print("input nuniques: ", len(env_indices))
            #print("nsplits: ", nsplits)
            #env_indices_splits = get_indices(env_indices, nsplits)
            env_indices_splits = get_groups(extra_info, env_indices, nsplits)
            print(env_indices_splits)

            # run para cmp
            ret = para(delayed(new_unique_chem_envs)(extra_info, chem_envs_groups, env_indices) for env_indices in env_indices_splits)

            et = time.time()
            print("split ", nsplits, " for count ", icount, " time: ", et - st)

            # run final cmp
            env_indices = []
            for x in ret:
                env_indices.extend([a[1] for a in x])
            env_indices = sorted(env_indices)
            print("output nuniques: ", len(env_indices))

            #print("--- extra info ---")
            #for i, x in enumerate(extra_info):
            #    if i in env_indices:
            #        print(i, sorted(x["others"]))
            
            # check diff
            temp_env_indices = env_indices.copy()
            for e_idx in temp_env_indices:
                compared_others = sorted([e_idx] + extra_info[e_idx]["others"])
                if set(compared_others).intersection(set(temp_env_indices)) != set(temp_env_indices):
                    break
            else:
                print("finished...")
                break
                
            if (icount+1) % 50 == 0:
                print("save temp comparasion info...")
                with open(directory/(saved_name+"-"+str(icount+1)), "wb") as fopen:
                    pickle.dump(extra_info, fopen)
            
        #unique = new_unique_chem_envs(extra_info, chem_envs_groups, env_indices)
        #et = time.time()
        #print("complete time: ", et - st)
    
    with open(saved_info, "wb") as fopen:
        pickle.dump(extra_info, fopen)

    st = time.time()

    unique_indices, unique_envs, unique_groups = merge_results(extra_info, chem_envs_groups, metadata)
        
    et = time.time()
    print("merge data time: ", et - st)

    return unique_envs, unique_groups

if __name__ == "__main__":
    ...