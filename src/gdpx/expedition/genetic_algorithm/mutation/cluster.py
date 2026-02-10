import copy
import functools
import itertools
from typing import Optional

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator

from gdpx.geometry.bounce import get_a_random_direction
from gdpx.geometry.spatial import check_atomic_distances
from gdpx.graph.base import AtomicGraph


class ClusterRattleMutation(OffspringCreator):
    """The cluster rattle mutation perturbs the position of a cluster of atoms based on graph theory."""

    def __init__(
        self,
        bond_distance_dict: dict[tuple[int, int], float],
        max_disp: float = 1.6,
        rattle_ratio: float = 0.4,
        graph_neigh_ratio: float = 1.03,
        covalent_ratio: tuple[float, float] = (0.8, 2.0),
        max_attempts: int = 100,
        num_muts: int = 1,
        use_tags: bool = True,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """"""
        super().__init__(num_muts=num_muts)
        self.descriptor = "ClusRatMutation"
        self.min_inputs = 1

        self.max_disp = max_disp
        self.rattle_ratio = rattle_ratio

        self.graph_neigh_ratio = graph_neigh_ratio

        self.bond_distance_dict = bond_distance_dict
        self.covalent_ratio = covalent_ratio

        self.max_attempts = max_attempts

        self.use_tags = use_tags

        self.rng = rng

        return

    def get_new_individual(self, parents: list[Atoms]):
        """"""
        f = parents[0]

        indi, extra_info = self.mutate(f)
        if indi is None:
            ...
        else:
            indi = self.initialize_individual(f, indi)
            indi.info["data"]["parents"] = [f.info["confid"]]
            indi.info["key_value_pairs"]["origin"] = self.descriptor + "_" + extra_info.split()[0]

        return indi, f"mutation: cluster_rattle {extra_info}"

    def mutate(self, atoms: Atoms) -> tuple[Optional[Atoms], str]:
        """"""
        mutant = copy.deepcopy(atoms)

        # Find clusters by graph theory
        group_indices = [i for i, tag in enumerate(mutant.get_tags()) if tag > 0]
        num_group_atoms = len(group_indices)
        # assert num_group_atoms > 0, "No tagged atoms found for cluster rattle mutation."

        check_geometry_func = functools.partial(
            check_atomic_distances,
            covalent_ratio=self.covalent_ratio,  # type: ignore[arg-type]
            bond_distance_dict=self.bond_distance_dict,
            allow_isolated=False,
        )

        if num_group_atoms > 0:
            graph_builder = AtomicGraph(mutant, graph_type="partial")
            graph_builder.build(group_indices=group_indices, ratio=self.graph_neigh_ratio)
            clusters = graph_builder.get_clusters(rebuild=True)

            num_clusters = len(clusters)
            assert num_clusters > 0, "No clusters found for cluster rattle mutation."
            num_rattled = max(1, int(num_clusters * self.rattle_ratio))
            num_rattled = min(num_clusters, num_rattled)
            selected_cluster_indices = self.rng.choice(num_clusters, size=num_rattled, replace=False)

            num_success = 0
            for i in selected_cluster_indices:
                cluster = clusters[i]
                prev_positions = copy.deepcopy(cluster.positions)
                host_indices = cluster.info["_host_indices"]
                intra_bond_pairs = list(itertools.permutations(host_indices, 2))
                for _ in range(self.max_attempts):
                    direction = get_a_random_direction(rng=self.rng)
                    displacement = direction / np.linalg.norm(direction) * self.max_disp  # TODO: wrap by cell?
                    mutant.positions[host_indices] = cluster.positions + displacement
                    if check_geometry_func(
                        mutant,
                        atomic_indices=host_indices,
                        excluded_pairs=intra_bond_pairs,
                    ):
                        num_success += 1
                        break
                    else:
                        mutant.positions[host_indices] = prev_positions

            extra_info = f"N{num_clusters}_R[{num_success}/{num_rattled}]"
        else:
            mutant = None
            extra_info = "No_Clusters"

        return mutant, extra_info
