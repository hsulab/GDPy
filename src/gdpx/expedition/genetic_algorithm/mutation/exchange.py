import copy
import functools
from typing import Any, Optional, Union

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator

from gdpx.geometry.composition import convert_string_to_adsorbate, convert_string_to_atoms
from gdpx.geometry.exchange import insert_one_particle, insert_one_particle_on_site, remove_one_particle
from gdpx.graph.adsorption import find_adsorption_sites_by_graph
from gdpx.nodes.region import RegionVariable
from gdpx.region.region import BaseRegion
from gdpx.utils.atoms_tags import get_tags_per_species


class ExchangeMutation(OffspringCreator):
    """The exchange mutation inserts or removes particles from the given structure."""

    def __init__(
        self,
        species: Union[str, list[str]],
        bond_distance_dict: dict[tuple[int, int], float],
        covalent_ratio: tuple[float, float] = (0.8, 2.0),
        num_min_max: Optional[Union[tuple[float, float], list[tuple[float, float]]]] = None,
        region: Optional[dict] = None,
        anchors: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        nsel: int = 1,
        num_muts: int = 1,
        use_tags: bool = True,
        max_attempts: int = 1000,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """Initialise the exchange mutation.

        Args:
            species: The chemical formulae of the species to be exchanged.
            bond_distance_dict: The bond distance dictionary.
            covalent_ratio: The covalent ratio for bond distance checking.
            num_min_max: The minimum and maximum number of each species allowed in the structure.
            region: The region where the exchange can occur.
            anchors: The predefined adsorption sites for each species.
            nsel: The number of sites to consider when using tags.
            num_muts: The number of mutations to perform.
            use_tags: Whether to use tags to identify species.
            max_attempts: The maximum number of attempts to insert/remove a particle.
            rng: The random number generator.

        """
        super().__init__(num_muts=num_muts)
        self.descriptor = "ExMutation"
        self.min_inputs = 1

        region = region if region is not None else {}
        if not isinstance(region, BaseRegion):
            self.region = RegionVariable(**region).value
        else:
            self.region = region

        self.bond_distance_dict = bond_distance_dict

        self.covalent_ratio = covalent_ratio

        self.rng = rng

        self.nsel = nsel

        self.use_tags = use_tags

        self.max_attempts = max_attempts

        if isinstance(species, str):
            self.species = [species]
        else:  # assume it is a list of chemical formulae
            self.species = species
        num_species = len(self.species)

        if num_min_max is None:
            _num_min_max = [(0.0, np.inf)] * num_species
        else:
            if isinstance(num_min_max, list):
                if isinstance(num_min_max[0], list):
                    _num_min_max = [(nm[0], nm[1]) for nm in num_min_max]
                else:
                    _num_min_max = [(num_min_max[0], num_min_max[1])] * num_species
            else:
                raise Exception(f"num_min_max `{num_min_max}` must be a list of tuples.")
        assert len(_num_min_max) == num_species
        assert all(isinstance(nm, tuple) and len(nm) == 2 for nm in _num_min_max), (
            "Each entry in num_min_max must be a tuple of (min, max)."
        )

        self.num_min_max = []
        for n_min, n_max in _num_min_max:
            if n_min is None:
                n_min = 0.0
            assert isinstance(n_min, float)
            if n_max is None:
                n_max = np.inf
            assert isinstance(n_max, float)
            if n_min >= n_max:
                raise Exception(f"n_min={n_min} >= n_max={n_max}")
            self.num_min_max.append((n_min, n_max))

        # Use predefined adsorption sites
        if anchors is not None:
            if isinstance(anchors, list):
                self.anchors = anchors
                num_anchors = len(anchors)
                assert num_anchors == num_species, f"num_anchors {num_anchors} != num_species {num_species}"
            else:
                self.anchors = [anchors] * num_species
        else:
            self.anchors = None

        # Get atoms objects for each species
        if self.anchors is not None:
            self._species_instances = {}
            for s, a in zip(self.species, self.anchors):
                if a is not None:
                    self._species_instances[s] = convert_string_to_adsorbate(s)
                else:
                    self._species_instances[s] = convert_string_to_atoms(s)
        else:
            self._species_instances = {s: convert_string_to_atoms(s) for s in self.species}

        return

    def get_new_individual(self, parents: list[Atoms]):
        """"""
        f = parents[0]

        indi, extra_info = self.mutate(f)
        if indi is None:
            return indi, "mutation: exchange"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info["confid"]]

        # finalize_individual, add sub operation descriptor
        indi.info["key_value_pairs"]["origin"] = self.descriptor + "_" + extra_info.split()[0]

        return indi, f"mutation: exchange {extra_info}"

    def mutate(self, atoms: Atoms):
        """"""
        mutant = copy.deepcopy(atoms)

        identities = get_tags_per_species(mutant)
        valid_identities = {}
        for k in self.species:
            v = identities.get(k, {})
            if len(v) > 0:
                valid_identities[k] = v

        # Check if the number of the selected species is within the tolerance
        species_to_exchange = str(self.rng.choice(self.species, replace=False))
        num_species_to_exchange = len(valid_identities.get(species_to_exchange, []))
        num_min_max = self.num_min_max[self.species.index(species_to_exchange)]

        if num_species_to_exchange <= num_min_max[0]:
            op = "insert"
        elif num_min_max[0] < num_species_to_exchange <= num_min_max[1]:
            op = self.rng.choice(["insert", "remove"], 1, replace=False)[0]
        else:
            op = "remove"

        # We should only remove species existing in the system
        if op == "remove":
            species_to_exchange = str(self.rng.choice(list(valid_identities.keys()), replace=False))
        else:
            ...

        # Run the exchange
        extra_info = ""
        if op == "insert":
            if self.anchors is None:
                # Insert the particle at a random position in the 3D space
                mutant, extra_info = insert_one_particle(
                    mutant,
                    self._species_instances[species_to_exchange],
                    region=self.region,
                    covalent_ratio=self.covalent_ratio,
                    bond_distance_dict=self.bond_distance_dict,
                    max_attempts=self.max_attempts,
                    rng=self.rng,
                )
            else:
                # Insert the particle at predefined adsorption sites
                site_params = self.anchors[self.species.index(species_to_exchange)]
                site_group_expr = site_params.get("group")
                assert isinstance(site_group_expr, str), (
                    "group expression must be provided for adsorption site finding."
                )
                find_sites_func = functools.partial(
                    find_adsorption_sites_by_graph,
                    group_expr=site_group_expr,
                    cutoff=site_params.get("cutoff", 3.0),
                    max_order=site_params.get("max_order", 3),
                    surf_index=site_params.get("surf_index", 2),
                )
                particle = self._species_instances[species_to_exchange]
                anchor_mode = particle.info.get("anchor_mode")
                if anchor_mode in ("mono", "bi"):
                    mutant, extra_info = insert_one_particle_on_site(
                        mutant,
                        particle,
                        find_sites_func=find_sites_func,
                        covalent_ratio=self.covalent_ratio,
                        bond_distance_dict=self.bond_distance_dict,
                        max_attempts=self.max_attempts,
                        rng=self.rng,
                    )
                else:
                    raise Exception(f"Unknown anchor_mode `{anchor_mode}` should not happen.")
        elif op == "remove":
            mutant, extra_info = remove_one_particle(mutant, valid_identities, species_to_exchange, rng=self.rng)
        else:
            ...  # Should not be here.

        return mutant, extra_info
