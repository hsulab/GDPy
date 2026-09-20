import itertools
from typing import Optional

import numpy as np
from ase import Atoms

from gdpx.exploration.persist.database import GlobalOptimisationDatabase as GODB
from gdpx.utils.atoms_tags import get_tags_per_species


from ...population.pool import (CandidatePool, compute_population_fitness, selection_weights,
                                count_looks_like, cache_fingerprints, delete_fingerprints)


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
        data_connection: GODB,
        population_size: int,
        comparator=None,
        use_extinct: bool = False,
        rng=None,
        print_func=print,
        debug_func=print,
    ):
        """"""
        self._print = print_func
        self._debug = debug_func

        self.dc = data_connection
        self.pop_size = population_size
        if comparator is None:
            from ...population.comparators import create_population_comparator

            comparator = create_population_comparator({"method": "interatomic_distance"})
        self.comparator = comparator
        self.use_extinct = use_extinct
        self.rng = np.random.default_rng() if rng is None else rng

        self.pop = []
        self.pairs: Optional[list[tuple[int, int]]] = None
        self.all_cand = None

        self.__initialise_population__()

        return

    def __initialise_population__(self) -> None:
        """Private method that initialises the population when the population is created."""
        pool = CandidatePool(self.dc, self.pop_size, self.comparator, self.use_extinct)
        self.pop = pool.candidates
        self.all_cand = pool.all_candidates
        self.__calc_participation__()

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

    def get_current_population(self) -> list[Atoms]:
        """Return borrowed candidates; callers copy at the mutation boundary."""
        return list(self.pop)

    def _select_candidates(self, candidates, size, with_history):
        if len(candidates) < size:
            return None
        indices = self.rng.choice(len(candidates), size=size, replace=False,
                                  p=selection_weights(candidates, with_history))
        return [candidates[i] for i in indices]

    def get_two_candidates(self, with_history=True) -> Optional[tuple[Atoms, Atoms]]:
        """Borrow two distinct parents, weighted by fitness and pairing history."""
        selected = self._select_candidates(self.pop, 2, with_history)
        return tuple(selected) if selected is not None else None

    def get_one_candidate(self, with_history=True) -> Optional[Atoms]:
        """Borrow one parent; the caller owns any subsequent mutation copy."""
        selected = self._select_candidates(self.pop, 1, with_history)
        return selected[0] if selected is not None else None


def group_structures_by_chemical_symbols(
    structures: list[Atoms],
) -> list[tuple[str, list[Atoms]]]:
    """Group structures by their chemical symbols.

    Note:
        We do not use chemical_formula here as structures with different chemical
        compositions may have the same chemical formula.

    """
    tribes_ = {}
    for k, v in itertools.groupby(structures, key=lambda a: "".join(a.get_chemical_symbols())):
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
    tribes: list[tuple[str, list[Atoms]]], min_size: int, rng: np.random.Generator
) -> Optional[list[Atoms]]:
    """Select structures from a tribe based on number probability.

    Args:
        tribes: Severl tribes.
        min_size: The minimum number of structures in the tribe will be considered.
        rng: Random number generator.

    Returns:
        A list of Atoms or None if no tribe satisfies `min_size`.

    """
    weights = []
    for tribe in tribes:
        size = len(tribe[1])
        if size >= min_size:
            weights.append(1.0 / size**0.5)
        else:
            weights.append(0.0)
    weights = np.array(weights)

    wsum = np.sum(weights)
    if wsum > 0.0:
        weights = weights / np.sum(weights)

        tribe_indices = list(range(len(tribes)))
        selected_tribe_index = rng.choice(tribe_indices, size=1, replace=False, p=weights)[0]
        tribe_structures = tribes[selected_tribe_index][1]
    else:
        tribe_structures = None

    return tribe_structures


class PopulationWithVariableComposition(Population):
    def __initialise_population__(self) -> None:
        """Private method that initialises the population when the population is created."""
        super().__initialise_population__()

        # Get tribes and select one tribe (composition) to get two structures
        self.tribes = group_structures_by_chemical_symbols(self.pop)

        return

    def get_two_candidates(self, with_history=True) -> Optional[tuple[Atoms, Atoms]]:
        tribe = select_tribe_structures(self.tribes, min_size=2, rng=self.rng)
        selected = self._select_candidates(tribe or [], 2, with_history)
        return tuple(selected) if selected is not None else None

    def get_one_candidate(self, with_history=True) -> Optional[Atoms]:
        tribe = select_tribe_structures(self.tribes, min_size=1, rng=self.rng)
        selected = self._select_candidates(tribe or [], 1, with_history)
        return selected[0] if selected is not None else None
