import ase.data
from ase import Atoms
from ase.io import write
from joblib import Parallel, delayed

from gdpx.utils.profiler import CustomTimer

from .modifier import GraphModifier
from .utils import single_swap_species


class GraphSwapModifier(GraphModifier):
    name: str = "graph_swap"

    def __init__(
        self,
        species: str,
        target: str,
        group: str,
        substrates=None,
        gmax: tuple[int, int, int] = (2, 2, 0),
        ratio: float = 1.1,
        skin: float = 0.25,
        *args,
        **kwargs,
    ):
        """Insert an adsorbate on sites according to graph representation."""
        super().__init__(substrates=substrates, *args, **kwargs)

        if species not in ase.data.chemical_symbols:
            raise Exception(f"graph_remove only supports single atom removal, got `{species}`.")
        self.species = species

        self.group = group

        self.target = target

        # Graph-building parameters
        self.gmax = gmax
        self.ratio = ratio
        self.skin = skin

        return

    def _irun(self, substrates: list[Atoms]) -> list[Atoms]:
        """Swap an adsorbate with another species."""
        self._print("---run swap---")
        # Get chemical environments of selected species that may be swapped
        with CustomTimer(name="exchange-adsorbate", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_swap_species)(
                    a,
                    self.group,
                    self.species,
                    self.target,
                    gmax=self.gmax,
                    ratio=self.ratio,
                    skin=self.skin,
                    print_func=self._print,
                    debug_func=self._debug,
                )
                for a in substrates
            )

        ret_frames, ret_envs = [], []
        for i, (frames, envs) in enumerate(ret):  # type: ignore
            nenvs = len(envs)
            ret_envs.extend(envs)
            ret_frames.extend(frames)
            self._print(f"number of sites {nenvs} to exchange for substrate {i}.")

        write(self.directory / f"possible_frames.xyz", ret_frames)

        # Get unique structures among substrates.
        # If Zn atoms were to swap with Cr, the chem envs of the rest Zn atoms
        # are used to compare the structure difference.
        graph_params = dict(
            gmax=self.gmax,
            ratio=self.ratio,
            skin=self.skin,
        )
        created_frames = self._compare_structures(ret_frames, graph_params, self.group)

        return created_frames
