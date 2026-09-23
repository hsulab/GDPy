#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time

from joblib import delayed, Parallel

from ase.ga.data import DataConnection
from gdpx.utils.profiler import CustomTimer
from gdpx.expedition.genetic_algorithm.comparator.interatomic_distance import get_sorted_dist_list
from gdpx.expedition.genetic_algorithm.operators import InteratomicDistanceComparator


def get_population(candidates, pop_size, comparator):
    # Fill up the population with the self.pop_size most stable
    # unique candidates.
    pop = []

    st = time.time()
    i = 0
    while i < len(candidates) and len(pop) < pop_size:
        print(f"{i=}  {len(pop)=}")
        if i % 10 == 0:
            et = time.time()
            print(f"Elapsed time: {et - st:.2f} seconds")
        c = candidates[i]
        i += 1
        eq = False
        for a in pop:
            if comparator.looks_like(a, c):
                eq = True
                break
        if not eq:
            pop.append(c)

    return pop

database = DataConnection("./mydb.db")

comparator = InteratomicDistanceComparator()

with CustomTimer("get_all_relaxed_candidates"):
    ue = False
    all_cand = database.get_all_relaxed_candidates(use_extinct=ue)


num_candidates = len(all_cand)
print(f"{num_candidates=}")


# Get fingerprints
candidates = all_cand[:]
with CustomTimer("get_sorted_dist_list"):
    # fingerprints = [get_sorted_dist_list(c) for c in all_cand[:10]]
    fingerprints = Parallel(n_jobs=12)(delayed(get_sorted_dist_list)(c) for c in candidates)
    for atoms, fingerprint in zip(candidates, fingerprints):
        atoms.info["interatomic_distance"] = fingerprint


# Get population
with CustomTimer("get_population"):
    population = get_population(candidates, 50, comparator)


if __name__ == "__main__":
    ...
