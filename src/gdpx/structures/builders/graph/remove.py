from typing import Optional

import ase.data
from ase import Atoms
from ase.io import write
from joblib import Parallel, delayed

from gdpx.utils.profiler import CustomTimer

from .modifier import GraphModifier
from .utils import single_remove_species


class GraphRemoveModifier(GraphModifier):
    name: str = "graph_remove"

    def __init__(
        self,
        species: str,
        group: str,
        substrates: Optional[list[Atoms]] = None,
        gmax: tuple[int, int, int] = (2, 2, 0),
        ratio: float = 1.1,
        skin: float = 0.25,
        *args,
        **kwargs,
    ):
        """Remove a species on sites according to graph representation."""
        super().__init__(substrates=substrates, *args, **kwargs)

        if species not in ase.data.chemical_symbols:
            raise Exception(f"graph_remove only supports single atom removal, got `{species}`.")
        self.species = species

        self.group = group

        # Graph-building parameters
        self.gmax = gmax
        self.ratio = ratio
        self.skin = skin

        return

    def _irun(self, substrates: list[Atoms]) -> list[Atoms]:
        """Remove atoms/molecules/adsorbates."""
        self._print("---run remove---")
        # Get chemical environments of selected species that may be removed
        with CustomTimer(name="remove-species", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_remove_species)(
                    a,
                    self.group,
                    self.species,
                    gmax=self.gmax,
                    ratio=self.ratio,
                    skin=self.skin,
                    print_func=self._print,
                    debug_func=self._debug,
                )
                for _, a in enumerate(substrates)
            )

        ret_frames, ret_envs = [], []
        for i, (frames, envs) in enumerate(ret):  # type: ignore
            nenvs = len(envs)
            ret_envs.extend(envs)
            ret_frames.extend(frames)
            self._print(f"number of sites {nenvs} to remove for substrate {i}.")

        write(self.directory / f"possible_frames.xyz", ret_frames)

        # Get unique structures among substrates.
        # If O atoms were to remove, the chem envs of the rest O atoms
        # are used to compare the structure difference.
        graph_params = dict(
            gmax=self.gmax,
            ratio=self.ratio,
            skin=self.skin,
        )
        created_frames = self._compare_structures(ret_frames, graph_params, self.group)

        return created_frames
