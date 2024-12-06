#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.ga.data import DataConnection

from gdpx.utils.atoms_tags import get_tags_per_species


def count_looks_like(a, all_cand, comp):
    """Utility method for counting occurrences."""
    n = 0
    for b in all_cand:
        if a.info["confid"] == b.info["confid"]:
            continue
        if comp.looks_like(a, b):
            n += 1
    return n


def compute_population_fitness(
    structures: List[Atoms], with_history=True
) -> List[float]:
    """Calculates the fitness."""
    scores = [x.info["key_value_pairs"]["raw_score"] for x in structures]
    min_s = min(scores)
    max_s = max(scores)
    T = min_s - max_s

    f = [0.5 * (1.0 - np.tanh(2.0 * (s - max_s) / T - 1.0)) for s in scores]
    if with_history:
        M = [float(atoms.info["n_paired"]) for atoms in structures]
        L = [float(atoms.info["looks_like"]) for atoms in structures]
        f = [
            f[i] * 1.0 / np.sqrt(1.0 + M[i]) * 1.0 / np.sqrt(1.0 + L[i])
            for i in range(len(f))
        ]

    return f


class Population:
    """This is a minimal implementation of the population class.

    The fitness with history is from the paper
    L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
    and the roulete wheel selection scheme described in
    R.L. Johnston Dalton Transactions,
    Vol. 22, No. 22. (2003), pp. 4193-4207

    """

    def __init__(
        self,
        data_connection: DataConnection,
        population_size: int,
        comparator=None,
        use_extinct: bool = False,
        rng=np.random.default_rng(),
    ):
        """"""
        self.dc = data_connection
        self.pop_size = population_size
        if comparator is None:
            from ase.ga.standard_comparators import AtomsComparator

            comparator = AtomsComparator()
        self.comparator = comparator
        self.use_extinct = use_extinct
        self.rng = rng

        self.pop = []
        self.pairs: Optional[List[Tuple[int, int]]] = None
        self.all_cand = None

        self.__initialise_population__()

        return

    def __initialise_population__(self) -> None:
        """Private method that initialises the population when the
        population is created."""
        # Get all relaxed candidates from the database
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(
            key=lambda x: x.info["key_value_pairs"]["raw_score"], reverse=True
        )

        # Fill up the population with the self.pop_size most stable
        # unique candidates.
        i = 0
        while i < len(all_cand) and len(self.pop) < self.pop_size:
            c = all_cand[i]
            i += 1
            eq = False
            for a in self.pop:
                if self.comparator.looks_like(a, c):
                    eq = True
                    break
            if not eq:
                self.pop.append(c)

        for a in self.pop:
            a.info["looks_like"] = count_looks_like(a, all_cand, self.comparator)

        self.all_cand = all_cand
        self.__calc_participation__()

        return

    def __calc_participation__(self) -> None:
        """Determines, from the database, how many times each
        candidate has been used to generate new candidates."""
        (participation, pairs) = self.dc.get_participation_in_pairing()
        for a in self.pop:
            if a.info["confid"] in participation.keys():
                a.info["n_paired"] = participation[a.info["confid"]]
            else:
                a.info["n_paired"] = 0
        self.pairs = pairs

        return

    def get_current_population(self) -> List[Atoms]:
        """Returns a copy of the current population."""
        return [a.copy() for a in self.pop]

    def get_two_candidates(self, with_history=True) -> Optional[Tuple[Atoms, Atoms]]:
        """Returns two candidates for pairing employing the
        fitness criteria.
        """
        if len(self.pop) < 2:
            return None

        fit = compute_population_fitness(self.pop, with_history=with_history)

        fmax = max(fit)
        c1 = self.pop[0]
        c2 = self.pop[0]
        used_before = False
        while c1.info["confid"] == c2.info["confid"] and not used_before:
            nnf = True
            while nnf:
                t = self.rng.integers(len(self.pop))
                if fit[t] > self.rng.random() * fmax:
                    c1 = self.pop[t]
                    nnf = False
            nnf = True
            while nnf:
                t = self.rng.integers(len(self.pop))
                if fit[t] > self.rng.random() * fmax:
                    c2 = self.pop[t]
                    nnf = False

            c1id = c1.info["confid"]
            c2id = c2.info["confid"]
            if self.pairs is not None:
                used_before = (min([c1id, c2id]), max([c1id, c2id])) in self.pairs
            else:
                raise Exception("This should not happen.")

        return (c1.copy(), c2.copy())

    def get_one_candidate(self, with_history=True) -> Optional[Atoms]:
        """Returns one candidate for mutation employing the
        fitness criteria.
        """
        c1 = None
        if len(self.pop) < 1:
            return c1

        fit = compute_population_fitness(self.pop, with_history=with_history)
        fmax = max(fit)
        nnf = True
        while nnf:
            t = self.rng.integers(len(self.pop))
            if fit[t] > self.rng.random() * fmax:
                c1 = self.pop[t]
                nnf = False

        if c1 is not None:
            c1 = copy.deepcopy(c1)
        else:
            ...

        return c1


def group_structures_by_chemical_symbols(
    structures: List[Atoms],
) -> List[Tuple[str, List[Atoms]]]:
    """Group structures by their chemical symbols.

    Note:
        We do not use chemical_formula here as structures with different chemical
        compositions may have the same chemical formula.

    """
    tribes_ = {}
    for k, v in itertools.groupby(
        structures, key=lambda a: "".join(a.get_chemical_symbols())
    ):
        if k in tribes_:
            tribes_[k].extend(list(v))
        else:
            tribes_[k] = list(v)

    tribes = []
    for k, v in tribes_.items():
        tags_dict = get_tags_per_species(v[0])
        name = " ".join([k + "_" + str(len(v)) for k, v in tags_dict.items()])
        tribes.append((name, v))

    return tribes


def select_tribe_structures(
    tribes: List[Tuple[str, List[Atoms]]], min_size: int, rng: np.random.Generator
) -> List[Atoms]:
    """"""
    weights = []
    for tribe in tribes:
        size = len(tribe[1])
        if size >= min_size:
            weights.append(1.0 / size**0.5)
        else:
            weights.append(0.0)
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    tribe_indices = list(range(len(tribes)))
    selected_tribe_index = rng.choice(tribe_indices, size=1, replace=False, p=weights)[
        0
    ]
    tribe_structures = tribes[selected_tribe_index][1]

    return tribe_structures


class PopulationWithVariableComposition(Population):

    def __initialise_population__(self) -> None:
        """Private method that initialises the population when the population is created."""
        # Get all relaxed candidates from the database
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(
            key=lambda x: x.info["key_value_pairs"]["raw_score"], reverse=True
        )

        # Fill up the population with the self.pop_size most stable
        # unique candidates.
        i = 0
        while i < len(all_cand) and len(self.pop) < self.pop_size:
            c = all_cand[i]
            i += 1
            eq = False
            for a in self.pop:
                if self.comparator.looks_like(a, c):
                    eq = True
                    break
            if not eq:
                self.pop.append(c)

        for a in self.pop:
            a.info["looks_like"] = count_looks_like(a, all_cand, self.comparator)

        self.all_cand = all_cand
        self.__calc_participation__()

        # Get tribes and select one tribe (composition) to get two structures
        self.tribes = group_structures_by_chemical_symbols(self.pop)

        return

    def get_two_candidates(self, with_history=True) -> Optional[Tuple[Atoms, Atoms]]:
        """Returns two candidates for pairing employing the fitness criteria."""
        assert hasattr(
            self, "tribes"
        ), "No tribes due to error from population initialisation."

        if len(self.pop) < 2:
            return None

        # Select a tribe
        tribe_structures = select_tribe_structures(
            self.tribes, min_size=2, rng=self.rng
        )
        num_structures_in_tribe = len(tribe_structures)

        # Pick two structures from the selected tribe
        fit = compute_population_fitness(tribe_structures, with_history=with_history)
        fmax = max(fit)
        c1 = tribe_structures[0]
        c2 = tribe_structures[0]
        used_before = False
        while c1.info["confid"] == c2.info["confid"] and not used_before:
            nnf = True
            while nnf:
                t = self.rng.integers(num_structures_in_tribe)
                if fit[t] > self.rng.random() * fmax:
                    c1 = tribe_structures[t]
                    nnf = False
            nnf = True
            while nnf:
                t = self.rng.integers(num_structures_in_tribe)
                if fit[t] > self.rng.random() * fmax:
                    c2 = tribe_structures[t]
                    nnf = False

            c1id = c1.info["confid"]
            c2id = c2.info["confid"]
            if self.pairs is not None:
                used_before = (min([c1id, c2id]), max([c1id, c2id])) in self.pairs
            else:
                raise Exception("This should not happen.")
        return (c1.copy(), c2.copy())

    def get_one_candidate(self, with_history: bool = True) -> Optional[Atoms]:
        """Returns one candidate employing the fitness criteria."""
        c1 = None
        if len(self.pop) < 1:
            return c1

        tribe_structures = select_tribe_structures(
            self.tribes, min_size=1, rng=self.rng
        )
        num_structures_in_tribe = len(tribe_structures)

        fit = compute_population_fitness(tribe_structures, with_history=with_history)
        fmax = max(fit)
        nnf = True
        while nnf:
            t = self.rng.integers(num_structures_in_tribe)
            if fit[t] > self.rng.random() * fmax:
                c1 = self.pop[t]
                nnf = False

        if c1 is not None:
            c1 = copy.deepcopy(c1)
        else:
            ...

        return c1


if __name__ == "__main__":
    ...
