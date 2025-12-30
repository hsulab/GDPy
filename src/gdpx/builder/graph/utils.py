import copy
from typing import Callable

import ase.data
import networkx as nx
from ase import Atoms

from gdpx.geometry.spatial import check_atomic_distances, get_bond_distance_dict
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


def single_insert_species(
    atoms: Atoms,
    group: str,
    species: Atoms,
    site: str,
    gmax: tuple[int, int, int],
    ratio: float,
    skin: float,
    print_func: Callable = print,
    debug_func: Callable = print,
) -> tuple[list[Atoms], list[nx.Graph]]:
    """Insert target species at atop sites.

    Currently, only monodentate adsorption on atop site is supported.

    TODO: bidentate/multidentate adsorption on bridge/hollow site.

    Args:
        atoms: The ASE Atoms object representing the structure.

    Returns:
        A list of structures with inserted atoms.
    """
    # Check we have valid adsorption sites
    group_indices = sorted(evaluate_group_expression(atoms, group))
    debug_func(f"group_indices to remove {group_indices}")

    chemical_symbols = atoms.get_chemical_symbols()
    for i in group_indices:
        if chemical_symbols[i] == site:
            break
    else:
        raise RuntimeError(f"There is no {site} to insert adsorbate by target group.")
    # Only check chemical environments for given site
    allowed_symbols = [site]
    group_indices = [i for i in group_indices if chemical_symbols[i] in allowed_symbols]

    # Get chemical environments from graph
    graph_builder = AtomicGraph(atoms, graph_type="expand", gmax=gmax)
    graph_builder.build(group_indices=group_indices, ratio=ratio, skin=skin)
    graph = graph_builder.graph
    assert isinstance(graph, nx.Graph)

    chem_envs = extract_chemical_environments(graph, atoms, group_indices, graph_radius=2)

    # Make sure only single atomic sites are found
    assert len(chem_envs) == len(group_indices), (
        "Single atoms group into one cluster. Try reducing the covalent radii."
    )

    # Find unique sites to remove for this structure
    unique_indices = get_unique_chemical_environments_by_bonds(chem_envs)
    unique_envs = [chem_envs[i] for i in unique_indices]

    # Check bond distance dict
    bond_distance_dict = get_bond_distance_dict(
        [ase.data.atomic_numbers[s] for s in set(chemical_symbols + species.get_chemical_symbols())]
    )

    # Create sctructures with inserted adsorbate
    unique_frames = []
    for g in unique_envs:
        for _, d in g.nodes.data():
            if d["central_ads"]:
                i = d["index"]
                chemical_symbol = chemical_symbols[i]
                if chemical_symbol == site:
                    new_atoms = copy.deepcopy(atoms)
                    new_species = copy.deepcopy(species)
                    contact_index = new_species.info["contact_index"]
                    contact_atom_symbol = new_species[contact_index].symbol  # type: ignore
                    bond_distance = (
                        ase.data.covalent_radii[ase.data.atomic_numbers[chemical_symbol]]
                        + ase.data.covalent_radii[ase.data.atomic_numbers[contact_atom_symbol]]
                    )
                    new_species.positions -= new_species.positions[contact_index]
                    new_species.positions += new_atoms.positions[i] + (0, 0, bond_distance)  # TODO: surface normal
                    new_atoms += new_species
                    if check_atomic_distances(
                        new_atoms,
                        covalent_ratio=(0.6, 2.0),  # type: ignore
                        bond_distance_dict=bond_distance_dict,
                    ):
                        unique_frames.append(new_atoms)
                    break
        else:
            # no valid adsorbate for this structure
            ...

    return unique_frames, unique_envs


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
        "Single atoms group into one cluster. Try reducing the covalent radii."
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


def single_swap_species(
    atoms: Atoms,
    group: str,
    species: str,
    target: str,
    gmax: tuple[int, int, int],
    ratio: float,
    skin: float,
    print_func: Callable = print,
    debug_func: Callable = print,
) -> tuple[list[Atoms], list[nx.Graph]]:
    """Exchange selected particles from the structure with target species.

    Currently, only single atom can be swapped.

    TODO: molecule.

    Args:
        atoms: The ASE Atoms object representing the structure.

    Returns:
        A list of structures with swappde atoms.

    """
    # Check if spec_indices are all species
    group_indices = evaluate_group_expression(atoms, group)
    debug_func(f"group_indices to remove {group_indices}")

    chemical_symbols = atoms.get_chemical_symbols()
    for i in group_indices:
        if chemical_symbols[i] == species:
            break
    else:
        raise RuntimeError(f"There is no {species} to swap by target group.")

    # Get chemical environments from graph
    graph_builder = AtomicGraph(atoms, graph_type="expand", gmax=gmax)
    graph_builder.build(group_indices=group_indices, ratio=ratio, skin=skin)
    graph = graph_builder.graph
    assert isinstance(graph, nx.Graph)

    chem_envs = extract_chemical_environments(graph, atoms, group_indices, graph_radius=2)

    # Make sure only single atoms are swapped
    assert len(chem_envs) == len(group_indices), (
        "Single atoms group into one cluster. Try reducing the covalent radii."
    )

    # Find unique sites to swap for this structure
    unique_indices = get_unique_chemical_environments_by_bonds(chem_envs)
    unique_envs = [chem_envs[i] for i in unique_indices]

    # Create sctructures with swapped species
    unique_frames = []
    for g in unique_envs:
        for _, d in g.nodes.data():
            if d["central_ads"]:
                i = d["index"]
                chemical_symbol = chemical_symbols[i]
                if chemical_symbol == species:
                    new_atoms = copy.deepcopy(atoms)
                    new_atoms[i].symbol = target  # type: ignore
                    unique_frames.append(new_atoms)
                    break
        else:
            # no valid species for this structure
            ...

    return unique_frames, unique_envs
