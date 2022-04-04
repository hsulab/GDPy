#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, random
from multiprocessing import Pool

import concurrent.futures

from GDPy.graph.utils import compare_chem_envs

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

def para_unique_chem_envs(chem_envs_groups, metadata=None, verbose=False, n_jobs=1):
    """"""
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
        unique_index_groups = [x[0] for x in unique]
        unique_envs = [x[1] for x in unique]
        # TODO: stop mpi when one proc found true
        cmp_results = Parallel(n_jobs=n_jobs)(delayed(compare_chem_envs)(env, unique_env) for unique_env in unique_envs)
        for idx2, cmp_ret in enumerate(cmp_results):
            if cmp_ret:
                unique_index_groups[idx2].append(index)
                break
        else:
            unique.append(([index], env))

        #for index2, (unique_indices, unique_env) in enumerate(unique):
        #    if verbose:
        #        print("Checking for uniqueness {:05d}/{:05d} {:05d}/{:05d}".format(index+1, len(chem_envs_groups), index2, len(unique)), end='\r')
        #    if compare_chem_envs(env, unique_env):
        #        unique_indices.append(index)
        #        break
        #else: # Was unique
        #    if verbose:
        #        print("")
        #    unique.append(([index], env))
    
    # Zip trick to split into two lists to return
    return zip(*[(env, [metadata[index] for index in indices]) for (indices, env) in unique])

def cmp_values(x):

    return x[0] == x[1]


def para_pool():
    cur_data = 1
    data = [100, 2, 3, 4, 5]

    st = time.time()

    pool = Pool(processes=4)
    #res = pool.apply_async(cmp_values, (10, 10,))
    #print(res.get(timeout=1))
    #for x in data:
    #    res = pool.apply_async(cmp_values, (cur_data, x))
    #cmp_ret = res.get()
    #if cmp_ret:
    #    pool.terminate()
    #    pool.join()
    it = pool.imap(cmp_values, [(cur_data,x) for x in data])
    while True:
        try:
            result = next(it)
            print("res: ", result)
            if result:
                pool.terminate()
                pool.join()
                break
        except StopIteration:
            break

    et = time.time()
    print("used times: ", et-st)

if __name__ == "__main__":
    cur_data = 1
    data = [1, 2, 3, 4, 5]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(cmp_values, (cur_data, x)): x for x in data}
        executor.shutdown(wait=False)
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
                #print('%r page is %d bytes' % (url, len(data)))
                pass