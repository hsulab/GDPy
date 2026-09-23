import numpy as np
import numpy.typing
from ase import Atoms
from ase.io import read, write

from gdpx.graph.base import AtomicGraph
from gdpx.group import evaluate_group_expression

from .describer import BaseDescriber


def preprocess_a_single_trajectory(frames: list[Atoms]):
    """"""

    return


class ClusterDescriber(BaseDescriber):
    """This class describes the cluster of a group of atoms in a structure."""

    name: str = "cluster"

    def __init__(self, group: str, *args, **kwargs):
        """Initialise the ClusterDescriber.

        Args:
            group: The group expression to evaluate.

        """
        super().__init__(*args, **kwargs)

        self.group = group

        return

    def run(self, structures, worker) -> numpy.typing.NDArray:
        """This method describes the cluster of a group of atoms in a structure."""
        self._print(f"{structures}")

        assert len(structures.shape) == 2, (
            f"Expected 2D array of shape (num_trajectories, num_snapshots) but got {structures.shape}."
        )

        new_trajectories = []
        for itraj, trajectory in enumerate(structures):
            new_traj_path = self.directory / f"traj-{itraj:>02d}-aligned.xyz"
            if not new_traj_path.exists():
                new_trajectory = []
                for iatoms, atoms in enumerate(trajectory):
                    group_indices = evaluate_group_expression(atoms, self.group)

                    graph_builder = AtomicGraph(atoms, graph_type="partial")
                    graph_builder.build(group_indices=group_indices, ratio=1.0, skin=0.2)
                    clusters = graph_builder.get_clusters(rebuild=True)
                    new_atoms = Atoms()
                    for cluster in clusters:
                        new_atoms += cluster
                    new_trajectory.append(new_atoms)
                    if iatoms % 200 == 0:
                        self._print(f"{iatoms:>12d}, num_clusters: {len(clusters)}")
                write(self.directory / f"traj-{itraj:>02d}-aligned.xyz", new_trajectory)
            else:
                self._print(f"Trajectory {itraj} already processed, skipping.")
                new_trajectory = read(new_traj_path, index=":")
            new_trajectories.append(new_trajectory)

        base_height = 9.2  # Ang
        cluster_heights = []
        for itraj, trajectory in enumerate(new_trajectories):
            new_cluster_heights = []
            for iatoms, atoms in enumerate(trajectory):
                new_cluster_heights.append(atoms.positions[:, 2].max() - base_height)
            cluster_heights.append(new_cluster_heights)

        cluster_heights = np.array(cluster_heights)
        np.save(self.directory / "clusters.npy", cluster_heights)

        return cluster_heights
