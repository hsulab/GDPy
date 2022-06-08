#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import networkx as nx
import multiprocessing
import concurrent.futures

from joblib import Parallel, delayed

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
plt.style.use("presentation")

# Handles isomorphism for bonds
bond_match = nx.algorithms.isomorphism.categorical_edge_match("bond", "")

# Handles isomorphism for atoms with regards to perodic boundary conditions
ads_match = nx.algorithms.isomorphism.categorical_node_match(["index", "ads"], [-1, False]) 

def node_symbol(atom, offset):
    return "{}:{}:[{},{},{}]".format(atom.symbol, atom.index, offset[0], offset[1], offset[2])

def unpack_node_name(node_symbol):
    #print("xxx: ", node_symbol)
    chem_sym, idx, offset = node_symbol.split(":")
    idx = int(idx)
    offset = eval(offset)

    return chem_sym, idx, offset

def bond_symbol(atoms, a1, a2):
    return "{}{}".format(*sorted((atoms[a1].symbol, atoms[a2].symbol)))

def grid_iterator(grid):
    """Yield all of the coordinates in a 3D grid as tuples

    Args:
        grid (tuple[int] or int): The grid dimension(s) to
                                  iterate over (x or (x, y, z))

    Yields:
        tuple: (x, y, z) coordinates
    """
    if isinstance(grid, int): # Expand to 3D grid
        grid = (grid, grid, grid)

    for x in range(-grid[0], grid[0]+1):
        for y in range(-grid[1], grid[1]+1):
            for z in range(-grid[2], grid[2]+1):
                yield (x, y, z)

def show_edges(graph):
    print("----- See Edges -----")
    for (u, v, d) in graph.edges.data():
        print(u, v, d)
    
    return
        
def show_nodes(graph):
    print("----- See Nodes -----")
    for (u, d) in graph.nodes.data():
        print(u, d)
    
    return

def show_components():
    #print("----- connected components -----")
    #for c in nx.connected_components(graph):
    #    print(c)
    pass

    return

def plot_graph(graph, fig_name="graph.png"):
    # plot graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.set_title("Graph")

    nx.draw(graph, with_labels=True)

    plt.savefig(fig_name)

    return


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

def pool_compare_chem_envs(chem_envs1, chem_envs2, pool):
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

def new_para_unique_chem_envs(chem_groups, metadata=None, n_jobs=1):
    """"""
    if metadata is None:
        metadata = [None] * len(chem_groups)
    # --- new para algo
    nframes = len(chem_groups)
    st_indices = list(range(nframes))
    end_indices = list(range(1,nframes)) + [0]
    pair_indices = list(zip(st_indices, end_indices))
    print(pair_indices)
    cmp_results = Parallel(n_jobs=n_jobs)(delayed(compare_chem_envs)(chem_groups[i], chem_groups[j]) for i, j in pair_indices)
    # create unique graph
    unique_graph = nx.Graph()
    for x in range(nframes):
        unique_graph.add_node(x)
    for (i, j), res in zip(pair_indices, cmp_results):
        if res:
            unique_graph.add_edge(i, j, look_like=res)
    #plot_graph(unique_graph, "ug.png")
    unique_group_graphs = [unique_graph.subgraph(c) for c in nx.connected_components(unique_graph)]

    unique_envs, unique_groups = [], []
    for g in unique_group_graphs:
        u_indices = [u for u in g]
        unique_groups.append([metadata[u] for u in u_indices])
        unique_envs.append(chem_groups[u_indices[0]])

    return unique_envs, unique_groups

def newnew_para_unique_chem_envs(chem_groups, metadata=None, n_jobs=1):
    """"""
    print("*** divide-conquer algo for chem_env cmp***")
    if metadata is None:
        metadata = [None] * len(chem_groups)
    # --- new para algo
    checked_indices = []

    nframes = len(chem_groups)
    unique_graph = nx.Graph()
    for x in range(nframes):
        unique_graph.add_node(x)

    with Parallel(n_jobs=n_jobs, prefer="processes") as para:
        for cur_idx in range(nframes):
            st = time.time()
            # check if already checked
            if cur_idx in checked_indices:
                continue

            # calc cmp
            end_indices = [x for x in range(cur_idx+1, nframes) if x not in checked_indices]
            st_indices = [cur_idx]*len(end_indices)
            pair_indices = list(zip(st_indices, end_indices))
            #print(pair_indices)
            cmp_results = para(delayed(compare_chem_envs)(chem_groups[i], chem_groups[j]) for i, j in pair_indices)

            # create unique graph
            new_checked = [cur_idx]
            for (i, j), res in zip(pair_indices, cmp_results):
                if res:
                    unique_graph.add_edge(i, j, dist=1)
                    new_checked.append(j)
            checked_indices.extend(new_checked)

            et = time.time()
            
            print("used time ", et-st, " for ", cur_idx, " for ", len(pair_indices))

            #local_graph = nx.node_connected_component(unique_graph, cur_idx)
            #new_checked = [u for u in local_graph if u not in checked_indices]
            #checked_indices.extend(new_checked)
            #print(f"--- {cur_idx} ---")
            #print([u for u in local_graph])
    #assert len(checked_indices) == nframes

    # merge results
    unique_group_graphs = [unique_graph.subgraph(c) for c in nx.connected_components(unique_graph)]

    unique_envs, unique_groups = [], []
    for g in unique_group_graphs:
        u_indices = sorted([u for u in g])
        unique_groups.append([metadata[u] for u in u_indices])
        unique_envs.append(chem_groups[u_indices[0]])

    return unique_envs, unique_groups




def para_unique_chem_envs(chem_envs_groups, metadata=None, verbose=False, n_jobs=1):
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
    print("*** using parallel chem-env cmp***")
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

    with Parallel(n_jobs=n_jobs) as para:
        for index, env in enumerate(chem_envs_groups):
            #unique_index_groups = [x[0] for x in unique]
            #unique_envs = [x[1] for x in unique]
            # TODO: stop mpi when one proc found true
            cmp_results = para(delayed(compare_chem_envs)(env, x[1]) for x in unique)
            for idx2, cmp_ret in enumerate(cmp_results):
                if cmp_ret:
                    unique[idx2][0].append(index)
                    break
            else:
                unique.append(([index], env))

    # Zip trick to split into two lists to return
    return zip(*[(env, [metadata[index] for index in indices]) for (indices, env) in unique])

def new_cmp_envs(a, b, idx):
    look_like = compare_chem_envs(a, b)

    return look_like, idx

class CmpWorker():
    def __init__(self, n_jobs):
        #self.pool = multiprocessing.Pool(processes=n_jobs)
        self.pool = multiprocessing.pool.ThreadPool(processes=n_jobs)
        self.unique_index = None

    def callback(self, result):
        if result[0]:
            #print("Solution found! Yay!")
            self.pool.terminate()
            self.unique_index = result[1]

    def do_job(self, env, unique):
        for i, x in enumerate(unique):
            self.pool.apply_async(
                new_cmp_envs, args=(env, x[1], i),
                callback=self.callback
            )

        self.pool.close()
        self.pool.join()

        return self.unique_index


def pool_para_unique_chem_envs(chem_envs_groups, metadata=None, verbose=False, n_jobs=1):
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
    print("*** using parallel chem-env cmp***")
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
        st = time.time()
        #unique_index_groups = [x[0] for x in unique]
        #unique_envs = [x[1] for x in unique]
        # TODO: stop mpi when one proc found true
        worker = CmpWorker(n_jobs)
        unique_index = worker.do_job(env, unique)
        if unique_index is not None:
            unique[unique_index][0].append(index)
        else:
            unique.append(([index], env))
        et = time.time()

        print("used time ", et-st, " for ", index, " for ", len(unique))

    # Zip trick to split into two lists to return
    return zip(*[(env, [metadata[index] for index in indices]) for (indices, env) in unique])

def newnew_cmp_envs(x):
    a, b, idx = x
    look_like = compare_chem_envs(a, b)

    return look_like, idx

def new_pool_para_unique_chem_envs(chem_envs_groups, metadata=None, verbose=False, n_jobs=1):
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
    print("*** using new pool parallel chem-env cmp***")
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

    pool = multiprocessing.pool.ThreadPool(n_jobs)
    for index, env in enumerate(chem_envs_groups):
        #unique_index_groups = [x[0] for x in unique]
        #unique_envs = [x[1] for x in unique]
        # TODO: stop mpi when one proc found true
        st = time.time()
        nunique = len(unique)
        if nunique > n_jobs*4:
            print("use Pool for ", index)
            unique_index = None
            #it = pool.imap(newnew_cmp_envs, [(env,x[1],u_idx) for u_idx, x in enumerate(unique)])
            #while True:
            #    try:
            #        result = next(it)
            #        if result[0]:
            #            unique_index = result[1]
            #            pool.terminate()
            #            break
            #    except StopIteration:
            #        break
            cur_uidx = 0
            exit_flag = True
            while exit_flag:
                end_uidx = cur_uidx + n_jobs*4
                if end_uidx > nunique:
                    end_uidx = nunique
                results = pool.map(newnew_cmp_envs, [(env,unique[u_idx][1],u_idx) for u_idx in range(cur_uidx, end_uidx)])
                #print(results)
                for res in results:
                    if res[0]:
                        unique_index = res[1]
                        print(res)
                        exit_flag = False
                        break
                else:
                    cur_uidx = end_uidx
                if end_uidx >= nunique:
                    exit_flag = False

            if unique_index is not None:
                unique[unique_index][0].append(index)
            else:
                unique.append(([index], env))
        else:
            for index2, (unique_indices, unique_env) in enumerate(unique):
                if compare_chem_envs(env, unique_env):
                    unique_indices.append(index)
                    break
            else: # Was unique
                unique.append(([index], env))
        et = time.time()

        print("used time ", et-st, " for ", index, " for ", len(unique))

    pool.close()
    pool.join()

    # Zip trick to split into two lists to return
    return zip(*[(env, [metadata[index] for index in indices]) for (indices, env) in unique])

def thread_para_unique_chem_envs(chem_envs_groups, metadata=None, verbose=False, n_jobs=1):
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
    print("*** using parallel chem-env cmp***")
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
        #unique_index_groups = [x[0] for x in unique]
        #unique_envs = [x[1] for x in unique]
        # TODO: stop mpi when one proc found true
        unique_index = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(newnew_cmp_envs, (env, x[1], u_idx)): u_idx for x, u_idx in enumerate(unique)}
            #executor.shutdown(wait=False)
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    res = future.result()
                    print("res: ", res)
                    if res:
                        break
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    pass
        pool = multiprocessing.Pool(n_jobs)
        it = pool.imap(newnew_cmp_envs, [(env,x[1],u_idx) for u_idx, x in enumerate(unique)])
        while True:
            try:
                result = next(it)
                if result[0]:
                    unique_index = result[1]
                    pool.terminate()
                    pool.join()
                    break
            except StopIteration:
                break

        if unique_index is not None:
            unique[unique_index][0].append(index)
        else:
            unique.append(([index], env))

    # Zip trick to split into two lists to return
    return zip(*[(env, [metadata[index] for index in indices]) for (indices, env) in unique])

def block_para_unique_chem_envs(chem_envs_groups, metadata=None, verbose=False, n_jobs=1):
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
    print("*** using parallel chem-env cmp***")
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
    batchsize = n_jobs
    print("batchsize: ", batchsize)

    nframes = len(chem_envs_groups)
    unique = [[[0], chem_envs_groups[0]]]

    with Parallel(n_jobs=n_jobs) as para:
        for index in range(1,nframes):
            env = chem_envs_groups[index]
            # --- find solution in a block
            nblocks = len(unique) // batchsize
            if nblocks == 0:
                nblocks += 1
            #print(nblocks)
            st_indices = [batchsize*iblock for iblock in range(nblocks)]
            end_indices = [x+batchsize for x in st_indices]
            end_indices[-1] = len(unique)

            #print(st_indices, end_indices)

            for st, end in zip(st_indices, end_indices):
                #print("---", st, end)
                # TODO: stop mpi when one proc found true
                cmp_results = para(delayed(compare_chem_envs)(env, x[1]) for x in unique[st:end])
                for idx2, cmp_ret in enumerate(cmp_results):
                    if cmp_ret:
                        unique[st+idx2][0].append(index)
                        break
                else:
                    unique.append(([index], env))

    # Zip trick to split into two lists to return
    return zip(*[(env, [metadata[index] for index in indices]) for (indices, env) in unique])


if __name__ == "__main__":
    paragroup_unique_chem_envs(range(103), [], n_jobs=8)
    pass