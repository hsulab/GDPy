import copy
from typing import Callable, Optional, Tuple

import ase.data
import networkx as nx
from ase import Atoms
from ase.io import write
from joblib import Parallel, delayed

from gdpx.graph.base import AtomicGraph
from gdpx.graph.expand import extract_chemical_environments, get_unique_chemical_environments_by_bonds
from gdpx.group import evaluate_group_expression
from gdpx.utils.profiler import CustomTimer

from .modifier import DEFAULT_GRAPH_PARAMS, GraphModifier


def single_remove_adsorbate(
    atoms: Atoms,
    group: str,
    species: str,
    gmax: Tuple[int, int, int],
    ratio: float,
    skin: float,
    print_func: Callable = print,
    debug_func: Callable = print,
) -> Tuple[list[Atoms], list[nx.Graph]]:
    """Remove selected particles from the structure.

    Currently, only single atom can be removed.

    TODO: molecule.

    Args:
        atoms: The ASE Atoms object representing the structure.

    Returns:
        A list of structures with removed atoms.

    """
    # Check if spec_indices are all species
    group_indices = sorted(evaluate_group_expression(atoms, group))
    debug_func(f"group_indices to remove {group_indices}")

    chemical_symbols = atoms.get_chemical_symbols()
    for i in group_indices:
        if chemical_symbols[i] != species:
            raise RuntimeError("Species to remove is inconsistent for those by indices.")

    # Get chemical environments from graph
    graph_builder = AtomicGraph(atoms, graph_type="expand", gmax=gmax)
    graph_builder.build(group_indices=group_indices, ratio=ratio, skin=skin)
    graph = graph_builder.graph
    assert isinstance(graph, nx.Graph)

    chem_envs = extract_chemical_environments(graph, atoms, group_indices, graph_radius=2)

    # Make sure only single atoms are removed
    assert len(chem_envs) == len(group_indices), (
        "Single atoms group into one adsorbate. Try reducing the covalent radii."
    )

    # Find unique sites to remove for this structure
    unique_indices = get_unique_chemical_environments_by_bonds(chem_envs)
    unique_envs = [chem_envs[i] for i in unique_indices]

    # Create sctructures with removed adsorbate
    unique_frames = []
    for g in unique_envs:
        for _, d in g.nodes.data():
            if d["central_ads"]:
                i = d["index"]
                chemical_symbol = chemical_symbols[i]
                if chemical_symbol == species:
                    new_atoms = copy.deepcopy(atoms)
                    del new_atoms[i]
                    unique_frames.append(new_atoms)
                    break
        else:
            # no valid adsorbate for this structure
            ...

    return unique_frames, unique_envs


class GraphRemoveModifier(GraphModifier):
    name: str = "graph_remove"

    def __init__(
        self,
        species: str,
        group: str,
        substrates: Optional[list[Atoms]] = None,
        graph: dict = DEFAULT_GRAPH_PARAMS,
        gmax: Tuple[int, int, int] = (2, 2, 0),
        ratio: float = 1.1,
        skin: float = 0.25,
        *args,
        **kwargs,
    ):
        """Remove an adsorbate on sites according to graph representation."""
        super().__init__(substrates=substrates, *args, **kwargs)

        if species not in ase.data.chemical_symbols:
            raise Exception(f"graph_remove only supports single atom removal, got `{species}`.")
        self.species = species

        self.group = group

        self.graph_params = graph

        # Graph-building parameters
        self.gmax = gmax
        self.ratio = ratio
        self.skin = skin

        return

    def _irun(self, substrates: list[Atoms]) -> list[Atoms]:
        """Remove atoms/molecules/adsorbates."""
        self._print("---run remove---")
        graph_params = copy.deepcopy(self.graph_params)
        graph_params.update(
            adsorbate_elements=[self.species],
        )

        # Get chemical environments of selected species that may be removed
        with CustomTimer(name="remove-adsorbate", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_remove_adsorbate)(
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
        created_frames = self._compare_structures(ret_frames, graph_params, self.group)

        return created_frames
