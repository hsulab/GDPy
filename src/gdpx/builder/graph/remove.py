import copy
from typing import Callable, Optional, Tuple

import networkx as nx
from ase import Atoms
from ase.io import write
from joblib import Parallel, delayed

from gdpx.graph.comparison import get_unique_environments_based_on_bonds
from gdpx.graph.creator import StruGraphCreator, extract_chem_envs
from gdpx.graph.utils import unpack_node_name
from gdpx.group import evaluate_group_expression
from gdpx.utils.profiler import CustomTimer

from .modifier import DEFAULT_GRAPH_PARAMS, GraphModifier


def single_remove_adsorbate(
    species: str,
    graph_params: dict,
    group: str,
    atoms: Atoms,
    print_func: Callable = print,
    debug_func: Callable = print,
) -> Tuple[list[Atoms], list[nx.Graph]]:
    """Remove selected particles from the structure.

    Currently, only single atom can be removed.

    TODO: molecule.

    Args:
        graph_params: Parameters for creating graphs.
        spec_params: Parameters for finding species to remove.

    """
    # Create graph from structure
    stru_creator = StruGraphCreator(**graph_params)

    # Check if spec_indices are all species
    group_indices = sorted(evaluate_group_expression(atoms, group))
    debug_func(f"group_indices to remove {group_indices}")

    chemical_symbols = atoms.get_chemical_symbols()
    for i in group_indices:
        if chemical_symbols[i] != species:
            raise RuntimeError("Species to remove is inconsistent for those by indices.")

    # Get chemical environments from graph
    graph = stru_creator.generate_graph(atoms, ads_indices=group_indices)
    chem_envs = extract_chem_envs(graph, atoms, group_indices, stru_creator.graph_radius)

    # Make sure only single atoms are removed
    assert len(chem_envs) == len(group_indices), (
        "Single atoms group into one adsorbate. Try reducing the covalent radii."
    )

    # Find unique sites to remove for this structure
    unique_indices = get_unique_environments_based_on_bonds(chem_envs)
    unique_envs = [chem_envs[i] for i in unique_indices]

    # Create sctructures with removed adsorbate
    unique_frames = []
    for g in unique_envs:
        for u, d in g.nodes.data():
            if d["central_ads"]:
                chem_sym, idx, offset = unpack_node_name(u)
                if chem_sym == species:
                    new_atoms = atoms.copy()
                    del new_atoms[idx]
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
        spectators: list[str],
        group: str,
        substrates: Optional[list[Atoms]] = None,
        graph: dict = DEFAULT_GRAPH_PARAMS,
        *args,
        **kwargs,
    ):
        """Remove an adsorbate on sites according to graph representation."""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.species = species

        self.group = group

        self.spectators = spectators
        self.graph_params = graph

        return

    def _irun(self, substrates: list[Atoms]) -> list[Atoms]:
        """Remove atoms/molecules/adsorbates."""
        self._print("---run remove---")
        graph_params = copy.deepcopy(self.graph_params)
        adsorbate_elements = copy.deepcopy(self.spectators)
        graph_params.update(adsorbate_elements=adsorbate_elements)

        # Get chemical environments of selected species that may be removed
        with CustomTimer(name="remove-adsorbate", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_remove_adsorbate)(
                    self.species,
                    graph_params,
                    self.group,
                    a,
                    print_func=self._print,
                    debug_func=self._debug,
                )
                for idx, a in enumerate(substrates)
            )

            ret_frames, ret_envs = [], []
            for i, (frames, envs) in enumerate(ret):
                nenvs = len(envs)
                # TODO: add info since it may be lost in atoms.copy() function
                # for a in frames:
                #    a.info["subid"] = subid
                # -- add data
                ret_envs.extend(envs)
                ret_frames.extend(frames)
                self._print(f"number of sites {nenvs} to remove for substrate {i}.")
        # nsites = len(ret_frames)
        # self._print(f"Total number of chemical environments: {nsites}")

        # - further unique envs among different substrates
        #   only compare chemical environments
        # unique_indices = get_unique_environments_based_on_bonds(ret_envs)
        # created_frames = [ret_frames[i] for i in unique_indices]

        # Compare the graph of chemical environments in the structure
        # If O atoms were to remove, the chem envs of the rest O atoms
        # are used to compare the structure difference.
        write(self.directory / f"possible_frames.xyz", ret_frames)

        # Get unique structures among substrates
        created_frames = self._compare_structures(ret_frames, graph_params, self.group)

        return created_frames
