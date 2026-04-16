import numpy as np
import numpy.typing
from ase import Atoms

from gdpx.group import evaluate_group_expression

from .describer import BaseDescriber


def stack_with_nan_padding(arrays):
    """"""
    max_shape = np.max([a.shape for a in arrays], axis=0)

    out_shape = (len(arrays),) + tuple(max_shape)
    out = np.full(out_shape, np.nan)

    for i, a in enumerate(arrays):
        slices = tuple(slice(0, s) for s in a.shape)
        out[(i,) + slices] = a

    return out


def get_colvar(atoms: Atoms, group_indices: list[int]) -> float:
    """"""
    # group_indices = evaluate_group_expression(atoms, group)

    # Compute the angle among the three atoms with the middle atom as the vertex
    if len(group_indices) != 3:
        return np.nan

    i, j, k = group_indices

    v1 = atoms[i].position - atoms[j].position
    v2 = atoms[k].position - atoms[j].position

    # normalize vectors
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cross = np.linalg.norm(np.cross(v1, v2))
    dot = np.dot(v1, v2)

    angle = np.degrees(np.arctan2(cross, dot))

    return angle


class ColvarDescriber(BaseDescriber):
    """This class describes the collective variable of a group of atoms in a structure."""

    name: str = "colvar"

    def __init__(self, colvars: list[dict], *args, **kwargs):
        """Initialise the ColvarDescriber.

        Args:
            group: The group expression to evaluate.

        """
        super().__init__(*args, **kwargs)

        self.colvars = colvars

        return

    def run(self, structures, worker) -> numpy.typing.NDArray:
        """This method describes the collective variable of a group of atoms in a structure."""
        self._print(f"{structures}")

        assert len(structures.shape) == 2, (
            f"Expected 2D array of shape (num_trajectories, num_snapshots) but got {structures.shape}."
        )

        cache_dirpath = self.directory / "cache"

        if not cache_dirpath.exists():
            cache_dirpath.mkdir(parents=True, exist_ok=True)
            # The input trajectories must be aligned and the colvars are computed on the aligned trajectories.
            all_results = []
            for itraj, trajectory in enumerate(structures):
                self._print(f"Processing trajectory {itraj:>02d}...")
                cache_path = cache_dirpath / f"traj-{itraj:>02d}-colvars.npy"
                if not cache_path.exists():
                    results = []
                    for iatoms, atoms in enumerate(trajectory):
                        if atoms is None:
                            break  # we may have padded Nones at the end of the trajectory, so we break when we encounter None
                        if iatoms % 2000 == 0:
                            self._print(f"{iatoms:>12d}")
                        cv_values = []
                        for colvar in self.colvars:
                            colvar_value = get_colvar(atoms, colvar["group"])
                            cv_values.append(colvar_value)
                        results.append(cv_values)
                else:
                    results = np.load(cache_path)
                results = np.array(results)
                np.save(cache_path, results)
                self._print(f"{results.shape=}")
                all_results.append(results)
        else:
            cache_paths = sorted(cache_dirpath.glob("traj-*-colvars.npy"))
            all_results = [np.load(cache_path) for cache_path in cache_paths]

        all_results = stack_with_nan_padding(all_results)

        self._print(f"{all_results.shape}")

        return all_results
