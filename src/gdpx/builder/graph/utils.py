import copy
from typing import Callable

import networkx as nx
from ase import Atoms

from gdpx.graph.base import AtomicGraph
from gdpx.graph.expand import extract_chemical_environments, get_unique_chemical_environments_by_bonds
from gdpx.group import evaluate_group_expression


def single_create_structure_graph(
    atoms: Atoms, group: str, gmax: tuple[int, int, int], ratio: float, skin: float
) -> list[nx.Graph]:
    """Create structure graph and get selected chemical environments.

    Find atoms with selected chemical symbols or in the defined region.

    Args:
        atoms: Input structure.

    Returns:
        A list of graphs that represent the chemical environments of selected atoms.

    """
    group_indices = evaluate_group_expression(atoms, group)
    graph_builder = AtomicGraph(atoms, graph_type="expand", gmax=gmax)
    graph_builder.build(group_indices=group_indices, ratio=ratio, skin=skin)
    graph = graph_builder.graph
    assert isinstance(graph, nx.Graph)

    chem_envs = extract_chemical_environments(graph, atoms, group_indices, graph_radius=2)

    return chem_envs


def single_remove_adsorbate(
    atoms: Atoms,
    group: str,
    species: str,
    gmax: tuple[int, int, int],
    ratio: float,
    skin: float,
    print_func: Callable = print,
    debug_func: Callable = print,
) -> tuple[list[Atoms], list[nx.Graph]]:
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
