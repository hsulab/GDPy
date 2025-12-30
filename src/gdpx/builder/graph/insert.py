import ase.data
from ase import Atoms
from ase.io import write
from joblib import Parallel, delayed

from gdpx.geometry.composition import convert_string_to_adsorbate
from gdpx.utils.profiler import CustomTimer

from .modifier import GraphModifier
from .utils import single_insert_species


class GraphInsertModifier(GraphModifier):
    name = "graph_insert"

    def __init__(
        self,
        species: str,
        site: str,
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

        self.species = species
        self._species_instance = convert_string_to_adsorbate(species)
        if self._species_instance.info.get("anchor_mode", None) != "mono":
            raise Exception(f"graph_insert only supports monodentate adsorbate insertion, got `{species}`.")

        if site not in ase.data.chemical_symbols:
            raise Exception(f"graph_insert only supports single atom site, got `{site}`.")
        self.site = site

        self.group = group

        # Graph-building parameters
        self.gmax = gmax
        self.ratio = ratio
        self.skin = skin

        return

    def _irun(self, substrates: list[Atoms]) -> list[Atoms]:
        """Insert an adsorabte on the substrate."""
        self._print("---run insert---")
        # Get chemical environments of selected species that may be removed
        with CustomTimer(name="insert-species", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_insert_species)(
                    a,
                    self.group,
                    self._species_instance,
                    self.site,
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
        # Using chemical environments of atoms in the group to compare.
        # If chemical symbols in adsorbates are not included in the group,
        # they will be ignored in the graph comparison.
        graph_params = dict(
            gmax=self.gmax,
            ratio=self.ratio,
            skin=self.skin,
        )
        created_frames = self._compare_structures(ret_frames, graph_params, self.group)

        return created_frames
