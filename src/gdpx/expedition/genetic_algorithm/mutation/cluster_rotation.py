import copy
import functools
import itertools
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator

from gdpx.geometry.spatial import check_atomic_distances
from gdpx.graph.base import AtomicGraph


def get_a_random_rotation_axis(rng: np.random.Generator) -> np.ndarray:
    """Get a random rotation axis."""
    u = rng.uniform(0.0, 1.0)
    v = rng.uniform(0.0, 1.0)

    phi = 2.0 * np.pi * u
    cos_theta = 2.0 * v - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    axis = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

    return axis


def get_rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Get a rotation matrix for rotating around a given axis by a given angle."""
    kx, ky, kz = axis
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1.0 - c

    rotation_matrix = np.array(
        [
            [kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [ky * kx * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s],
            [kz * kx * v - ky * s, kz * ky * v + kx * s, kz * kz * v + c],
        ]
    )

    return rotation_matrix


class ClusterRotationMutation(OffspringCreator):
    """The cluster rotation mutation rotates a cluster of atoms based on graph theory."""

    def __init__(
        self,
        bond_distance_dict: dict[tuple[int, int], float],
        max_angle: float = 360.0,
        rotation_axis: Optional[Union[str, np.ndarray]] = None,
        action_ratio: float = 0.4,
        graph_neigh_ratio: float = 1.03,
        covalent_ratio: tuple[float, float] = (0.8, 2.0),
        max_attempts: int = 100,
        num_muts: int = 1,
        use_tags: bool = True,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """"""
        super().__init__(num_muts=num_muts)
        self.descriptor = "ClusRotMutation"
        self.min_inputs = 1

        self.max_angle = max_angle
        self.action_ratio = action_ratio

        # Check if rotation_axis is x, y, z or a 3D vector,
        # otherwise, it will be set to a random rotation axis
        if isinstance(rotation_axis, str):
            if rotation_axis.lower() == "x":
                self.rotation_axis = np.array([1.0, 0.0, 0.0])
            elif rotation_axis.lower() == "y":
                self.rotation_axis = np.array([0.0, 1.0, 0.0])
            elif rotation_axis.lower() == "z":
                self.rotation_axis = np.array([0.0, 0.0, 1.0])
            else:
                raise Exception(f"Invalid rotation_axis string: {rotation_axis}. Must be 'x', 'y', or 'z'.")
        else:
            self.rotation_axis = rotation_axis

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

        return indi, f"mutation: cluster_rotation {extra_info}"

    def mutate(self, atoms: Atoms) -> tuple[Optional[Atoms], str]:
        """"""
        mutant = copy.deepcopy(atoms)
        extra_info = ""

        # Find clusters by graph theory
        group_indices = [i for i, tag in enumerate(mutant.get_tags()) if tag > 0]
        num_group_atoms = len(group_indices)
        # assert num_group_atoms > 0, "No tagged atoms found for cluster rotation mutation."

        check_geometry_func = functools.partial(
            check_atomic_distances,
            covalent_ratio=self.covalent_ratio,
            bond_distance_dict=self.bond_distance_dict,
            allow_isolated=False,
        )

        if num_group_atoms > 0:
            graph_builder = AtomicGraph(mutant, graph_type="partial")
            graph_builder.build(group_indices=group_indices, ratio=self.graph_neigh_ratio)
            clusters = graph_builder.get_clusters(rebuild=True)

            num_clusters = len(clusters)
            assert num_clusters > 0, "No clusters found for cluster rotation mutation."
            num_rotated = max(1, int(num_clusters * self.action_ratio))
            num_rotated = min(num_rotated, num_clusters)
            selected_cluster_indices = self.rng.choice(num_clusters, size=num_rotated, replace=False)

            num_success = 0
            for i in selected_cluster_indices:
                cluster = clusters[i]
                prev_positions = copy.deepcopy(cluster.positions)
                host_indices = cluster.info["_host_indices"]
                intra_bond_pairs = list(itertools.combinations(host_indices, 2))
                cluster_centre = cluster.get_center_of_mass()  # TODO: use cop?
                for _ in range(self.max_attempts):
                    # rotate cluster around z-axis by a random angle
                    degree = self.rng.uniform(0, self.max_angle)
                    theta = np.deg2rad(degree)
                    axis = (
                        self.rotation_axis
                        if self.rotation_axis is not None
                        else get_a_random_rotation_axis(rng=self.rng)
                    )
                    # build rotation matrix
                    rotation_matrix = get_rotation_matrix(axis, theta)
                    rel_pos = cluster.positions - cluster_centre
                    rot_pos = rel_pos @ rotation_matrix.T + cluster_centre
                    mutant.positions[host_indices] = rot_pos
                    if check_geometry_func(
                        mutant,
                        atomic_indices=host_indices,
                        excluded_pairs=intra_bond_pairs,
                    ):
                        num_success += 1
                        break
                    else:
                        mutant.positions[host_indices] = prev_positions

            extra_info = f"N{num_clusters}_R[{num_success}/{num_rotated}]"
        else:
            mutant = None
            extra_info = "No_Clusters"

        return mutant, extra_info
