from typing import Optional

from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed

from gdpx.graph.comparison import unique_chem_envs
from gdpx.utils.profiler import CustomTimer

from ..builder import StructureModifier
from .utils import single_create_structure_graph

DEFAULT_GRAPH_PARAMS = dict(
    pbc_grid=[2, 2, 0],
    graph_radius=2,
    neigh_params=dict(covalent_ratio=1.1, skin=0.25),
)


class GraphModifier(StructureModifier):
    def run(
        self,
        substrates: Optional[list[Atoms]] = None,
        size: int = 1,
        *args,
        **kwargs,
    ) -> list[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        prev_directory = self.directory
        curr_substrates = self.substrates

        for i in range(size):
            self.directory = prev_directory / f"graph-{i}"
            self.directory.mkdir(parents=True, exist_ok=True)
            cached_filepath = self.directory / "enumerated.xyz"
            if not cached_filepath.exists():
                self._print("-- run graph results --")
                modified_structures = self._irun(
                    curr_substrates,
                )
                write(self.directory / "enumerated.xyz", modified_structures)
            else:
                self._print("-- use cached results --")
                modified_structures = read(self.directory / "enumerated.xyz", ":")
            n_structures = len(modified_structures)
            self._print(f"nframes: {n_structures}")
            curr_substrates = modified_structures

        self.directory = prev_directory

        return modified_structures

    def _irun(self, substrates: list[Atoms]) -> list[Atoms]:
        """"""

        raise NotImplementedError()

    def _compare_structures(self, ret_frames: list[Atoms], graph_params: dict, group: str):
        """"""
        with CustomTimer(name="create-graphs", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_create_structure_graph)(a, group, **graph_params) for a in ret_frames
            )

        # Check if the ret is empty, it happens when all species are removed/exchanged...
        chemical_environments = []
        for x in ret:
            chemical_environments.extend(x)  # type: ignore

        if chemical_environments:
            ret_env_groups = ret
            self._print("Typical Chemical Environment " + str(chemical_environments[0]))
            with CustomTimer(name="check-uniqueness", func=self._print):
                _, unique_groups = unique_chem_envs(ret_env_groups, list(enumerate(ret_frames)))

            # Get unique structures
            created_frames = []
            for x in unique_groups:
                created_frames.append(x[0][1])
            num_candidates = len(created_frames)

            unique_data = []
            for i, x in enumerate(unique_groups):
                data = ["ug" + str(i)]
                data.extend([a[0] for a in x])
                unique_data.append(data)
            content = "# unique, indices\n"
            content += f"# ncandidates {num_candidates}\n"
            for d in unique_data:
                content += ("{:<8s}  " + "{:<8d}  " * (len(d) - 1) + "\n").format(*d)

            unique_info_path = self.directory / f"unique-info.txt"
            with open(unique_info_path, "w") as fopen:
                fopen.write(content)
        else:
            self._print("Cannot find valid species...")
            created_frames = ret_frames
            num_candidates = len(created_frames)

        return created_frames
