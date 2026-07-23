#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import functools
import time
from typing import Callable, Optional

import ase.data
import numpy as np
from ase import Atoms

from gdpx.geometry.composition import CompositionSpace
from gdpx.geometry.exchange import insert_one_particle_on_site
from gdpx.geometry.spatial import get_bond_distance_dict
from gdpx.graph.adsorption import find_adsorption_sites_by_graph
from gdpx.region.region import BaseRegion
from gdpx.utils.atoms_tags import sort_structures_by_natoms_per_type, sort_structures_by_tags

from .builder import StructureModifier


class AdsorbateInsertionModifier(StructureModifier):
    """Modifier to insert adsorbates onto a surface."""

    name: str = "adsorbate_insertion"

    def __init__(
        self,
        composition,
        sites,
        box=None,
        pbc: bool = True,
        use_tags: bool = True,
        covalent_ratio=[0.8, 2.0],
        molecular_distances=[None, None],
        custom_pair_distances: Optional[list[tuple[str, str, float]]] = None,
        max_times_size: int = 10,
        sort_by_tags: bool = True,
        sort_by_natoms_per_type: bool = True,
        type_list: Optional[list[str]] = None,
        *args,
        **kwargs,
    ):
        """Initialize the AdsorbateInsertionModifier."""
        super().__init__(*args, **kwargs)

        # Save init params
        self._init_params = dict(
            composition=composition,
            sites=sites,
            box=box,
            pbc=pbc,
            covalent_ratio=covalent_ratio,
            molecular_distances=molecular_distances,
            custom_pair_distances=custom_pair_distances,
            max_times_size=max_times_size,
            sort_by_tags=sort_by_tags,
            sort_by_natoms_per_type=sort_by_natoms_per_type,
            type_list=type_list,
            **kwargs,
        )

        # Overwrite substrates if it is a file path
        if self._input_substrates is not None:
            self._init_params["substrates"] = self._input_substrates

        # Additional types in the composition space
        self.type_list = type_list if type_list is not None else []

        # Check composition
        self._compspec = CompositionSpace(composition)

        # Check box
        try:
            if box is not None:
                box = np.array(box)
                if box.size == 3:
                    self.box = np.diag(box)
                else:  # assume it is (3,3)
                    self.box = np.reshape(box, (3, 3))
            else:
                self.box = None
        except:
            raise RuntimeError(f"box must be a (3,) or (3,3) array but `{box}` is given.")

        self.pbc = pbc

        # To compatible with GA engine
        self._substrate = None
        if self.substrates is not None:
            self._substrate = self.substrates[0]
        else:
            # Unlike random_structure, adsorbate_insertion must have substrates
            # self._substrate = Atoms("", cell=self.box, pbc=self.pbc)
            raise Exception("`adsorbate_insertion` must have substrates to be provided.")

        self.use_tags = use_tags
        if not self.use_tags:
            raise Exception("`adsorbate_insertion` must have use_tags to be True.")

        # Check sites
        if "group" in sites:
            # use the same site setting for all species
            self._sites = dict(_default=sites)
        else:
            # use per-species site setting
            for k in composition.keys():
                if k not in sites:
                    raise Exception(f"site settings for `{k}` must be provided in `{sites}`.")
            self._sites = sites

        # Spatial tolerance
        self.covalent_ratio = covalent_ratio

        if molecular_distances[0] is None:
            molecular_distances[0] = -np.inf
        if molecular_distances[1] is None:
            molecular_distances[1] = np.inf
        self.molecular_distances = molecular_distances

        self.custom_pair_distances = custom_pair_distances

        # Attempts
        self.max_times_size = max_times_size

        # Whether we should have a consistent tags
        self.sort_by_tags = sort_by_tags

        # Whether we should have structures order by natoms per elements
        self.sort_by_natoms_per_type = sort_by_natoms_per_type

        return

    def _infer_chemical_types_in_composition_space(self) -> list[str]:
        """"""
        chemical_symbols = self._compspec.get_chemical_symbols()
        if self.substrates is not None:
            for substrate in self.substrates:
                chemical_symbols.extend(substrate.get_chemical_symbols())
        chemical_symbols.extend(self.type_list)
        chemical_symbols = sorted(list(set(chemical_symbols)))

        return chemical_symbols

    def _infer_chemical_numbers_in_composition_space(self) -> list[int]:
        """Infer what chemical numbers may occur based on the composition space and the substrates.

        This is normally used to determine the covalent bond distances.
        """
        chemical_symbols = self._infer_chemical_types_in_composition_space()
        chemical_numbers = [ase.data.atomic_numbers[s] for s in chemical_symbols]

        return chemical_numbers

    def get_bond_distance_dict(self, ratio: float = 1.0) -> dict:
        """"""
        chemical_numbers = self._infer_chemical_numbers_in_composition_space()
        bond_distance_dict = get_bond_distance_dict(chemical_numbers, ratio=ratio)

        return bond_distance_dict

    def get_custom_pair_distance_dict(self) -> Optional[dict[tuple[int, int], float]]:
        """Get custom pair distance dictionary.

        Note:
            Atomic pair distances.

        Todo:
            This should extend to molecular distances.

        """
        custom_pair_distance_dict = {}
        if self.custom_pair_distances is not None:
            # map chemical symbols to numbers
            for (s0, s1, dist) in self.custom_pair_distances:
                n0 = ase.data.atomic_numbers[s0]
                n1 = ase.data.atomic_numbers[s1]
                custom_pair_distance_dict[(n0, n1)] = dist
                custom_pair_distance_dict[(n1, n0)] = dist
        else:
            ...

        return custom_pair_distance_dict

    def run(self, substrates: Optional[list[Atoms]] = None, size: int = 1, *args, **kwargs) -> list[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        if self.substrates is not None:
            ...
        else:
            raise Exception("`adsorbate_insertion` must have substrates to be provided.")

        # Infer chemical species may occur in structures
        bond_distance_dict = self.get_bond_distance_dict()

        custom_pair_distance_dict = self.get_custom_pair_distance_dict()

        # Build find adsorption sites function
        def build_find_sites_func(species: str) -> Callable:
            site_params = self._sites.get(species, self._sites.get("_default"))
            find_sites_func = functools.partial(
                find_adsorption_sites_by_graph,
                group_expr=site_params.get("group"),
                cutoff=site_params.get("cutoff", 3.0),
                max_order=site_params.get("max_order", 3),
                surf_index=site_params.get("surf_index", 2),
            )

            return find_sites_func

        # Generate structures
        frames = []
        for isub, substrate in enumerate(self.substrates):
            self._print(f"generating structures based on substrate-{isub:>04d}.")
            st = time.time()
            num_atoms_in_substrate = len(substrate)
            for _ in range(size):
                fragments = self._compspec.get_fragments_from_one_composition(
                    rng=self.rng,
                    use_ads=True,
                )
                candidate = copy.deepcopy(substrate)
                for particle in fragments:
                    _cand, _ = insert_one_particle_on_site(
                        candidate,
                        particle,
                        build_find_sites_func(particle.get_chemical_formula()),
                        covalent_ratio=self.covalent_ratio,
                        bond_distance_dict=bond_distance_dict,
                        custom_pair_distance_dict=custom_pair_distance_dict,
                        num_atoms_in_substrate=num_atoms_in_substrate,  # Make inter-adsorbate distance checked
                        sort_tags=self.sort_by_tags,
                        rng=self.rng,
                    )
                    if _cand is not None:
                        candidate = _cand
                frames.append(candidate)
            et = time.time()
            self._print(f"structures generated in {et-st:.2f} seconds.")

        # Sort atoms in each structure by tags
        if self.sort_by_natoms_per_type:
            chemical_types = self._infer_chemical_types_in_composition_space()
            frames = sort_structures_by_natoms_per_type(frames, chemical_types)

        # if self.sort_by_tags:
        #     frames = sort_structures_by_tags(frames)

        return frames

    def as_dict(self) -> dict:
        """"""
        params = dict(method=self.name)
        params.update(**copy.deepcopy(self._init_params))

        return params


if __name__ == "__main__":
    ...
