#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import dataclasses
import pathlib
from typing import Callable, Optional, List, Mapping
from itertools import chain

import numpy as np
import networkx as nx

from ase import Atoms
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList

from .. import config

from .utils import grid_iterator, node_symbol, bond_symbol, unpack_node_name

"""Create the graph representation of a structure based on its neighbour list.
"""

DEFAULT_NEIGH_PARAMS = dict(
    covalent_ratio = 1.0,
    skin = 0.0,
    self_interaction = False,
    bothways = True,
)

DIS_SURF2SURF = 2
DIS_ADS2SURF = 1

@dataclasses.dataclass
class GraphRadius:

    """Define some graph edge attributes."""

    radius: int = 2


class NeighGraphCreator():

    # Parameters for ASE neighbor list.
    covalent_ratio: float= 1.0 #: Multiplier for the covalent bond distance.
    skin = 0.0 # TODO: the neighbours will be in cutoff + skin
    self_interaction = False #: Whether consider self-interaction under PBC.
    bothways = True #: Whether return the full neighbor list.

    #: PBC grid.
    #pbc_grid: List[int] = [2, 2, 0] # z=0 is for surface systems.

    def __init__(
        self, covalent_ratio=1.0, skin=0.0, self_interaction=False, bothways=True, 
        *args, **kwargs
    ):
        """"""
        #for key, value in kwargs.items():
        #    if hasattr(self, key):
        #        setattr(self, key, value)
        self.covalent_ratio = covalent_ratio
        self.skin = skin
        self.self_interaction = self_interaction
        self.bothways = bothways

        return
    
    def build_neighlist(self, atoms, bothways=True):
        # NOTE: for debug only
        #content = "----- Build NeighborList -----\n"
        #content += ("mult: {} skin: {} bothways: {}\n").format(
        #    self.covalent_ratio, self.skin, bothways
        #)
        #print(content)

        nl = NeighborList(
            natural_cutoffs(atoms, mult=self.covalent_ratio), 
            skin = self.skin, sorted=False,
            self_interaction=self.self_interaction, 
            bothways = bothways
        )
        #print(natural_cutoffs(atoms, mult=self.covalent_ratio))

        return nl


class StruGraphCreator():

    """Create the graph representation of a structure based on its neighbour list.
    """
    #: Distance [Ang].
    substrate_adsorbate_distance: int = 2.5 

    #: Working directory.
    _directory = "./"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    def __init__(
        self, adsorbate_elements: List[str],
        pbc_grid: List[int]=[2,2,0], graph_radius: int=2, 
        neigh_params: dict=DEFAULT_NEIGH_PARAMS, 
        *args, **kwargs
    ):
        """Init graph creator.

        Args:
            adsorbate_elements: Elements that may be considered as adsorbates.
            graph_radius: 
                The subgraph size for comparasion. 
                # help='Sets the graph radius, this can be tuned for different behavior')

        """
        # NOTE: half neigh is enough for creating structure graph
        self.adsorbate_elements = adsorbate_elements
        # self.adsorbate_indices = adsorbate_indices

        self.graph_radius = graph_radius
        self.pbc_grid = pbc_grid

        # - neigh params
        # covalent_ratio = kwargs.pop("covalent_ratio", 1.0)
        # pbc_grid = kwargs.pop("pbc_grid", [2,2,0])
        # skin = kwargs.pop("skin", 0.0)
        # self_interaction = kwargs.pop("self_interaction", False)
        # bothways = kwargs.pop("bothways", True)
        # self.neigh_creator = NeighGraphCreator(
        #    covalent_ratio=covalent_ratio, skin=skin,
        #    self_interaction=self_interaction, bothways=bothways,
        #    pbc_grid=pbc_grid, *args, **kwargs
        # )
        self.neigh_creator = NeighGraphCreator(**neigh_params)

        return

    @property
    def directory(self):

        return self._directory

    @directory.setter
    def directory(self, directory_):
        """"""
        self._directory = pathlib.Path(directory_)

        return

    @property
    def DIS_SURF2SURF(self):

        return 2

    @property
    def DIS_ADS2SURF(self):

        return 1

    def add_atoms_node(self, graph, sym1, a1, o1, **kwargs):
        """Add an atom node to the graph.

        Args:
            graph: Graph.
            a1: Atom index.
            o1: Atom offset in PBC.

        """
        graph.add_node(
            node_symbol(sym1, a1, o1), index=a1, central_ads=False, **kwargs
        )

        return

    def add_atoms_edge(
        self, 
        graph,
        sym1, sym2,
        a1, a2, o1, o2, 
        dis,
        adsorbate_atoms, 
        **kwargs
    ):
        """ graph edge format
            dist 
                2 bond in substrate
                1 bond between adsorbate and substrate
                0 bond in adsorbate
            ads_only 
                0 means the bond is in the adsorbate
                2 otherwise
        """

        dist = self.DIS_SURF2SURF - (self.DIS_ADS2SURF if a1 in adsorbate_atoms else 0) - (self.DIS_ADS2SURF if a2 in adsorbate_atoms else 0)

        graph.add_edge(
            # node index
            node_symbol(sym1, a1, o1),
            node_symbol(sym2, a2, o2),
            # attributes - edge data
            bond = bond_symbol(sym1, sym2, a1, a2),
            index = "{}:{}".format(*sorted([a1, a2])),
            dist = dist,
            dist_edge = dis,
            ads_only = 0 if (a1 in adsorbate_atoms and a2 in adsorbate_atoms) else 2,
            **kwargs
        )

        return

    def check_system(
        self, atoms: Atoms, min_vacuum_ratio: float=0.3,
        substrate_indices: Optional[List[int]] = None
    ) -> str:
        """Determine system type.

        Args:
            atoms: Atoms object.
            substrate_indices: Atomic indices used to determine the system type.

        Return:
            A string that desribes the system type.

        """
        # TODO: solvent-solid interface

        cell_length = np.linalg.norm(atoms.get_cell(), axis=1)
        coord_range = np.max(atoms.positions, axis=0) - np.min(atoms.positions, axis=0)

        vacuum_ratio = (cell_length - coord_range) / cell_length
        num_vacuum = sum(vacuum_ratio >= min_vacuum_ratio)
        self._print(f"{vacuum_ratio = }")

        if num_vacuum == 3:
            system = "cluster"  # or molecule
        elif num_vacuum == 2:  # nanowire?
            # TODO: find direction?
            system = "wire"
        elif num_vacuum == 1:
            # TODO: find direction?
            system = "surface"
        else:  # num_vacuum == 0
            system = "bulk"

        return system

    def generate_graph(
        self, atoms: Atoms, ads_indices: List[int]=None, 
        clean_substrate: Atoms=None
    ) -> nx.Graph:
        """Generate a graph for the input atoms."""
        # - create graph with adsorbates
        full = self._generate_structure_graph(atoms, ads_indices)
        input_ads_indices = copy.deepcopy(ads_indices)

        # - create a graph of clean substrate plus adsorbate graph
        #   to avoid surface relaxation
        if clean_substrate is not None: # BUG: need an alignment operation
            self._print("use clean substrate plus adsorbates...")
            clean_graph = self._generate_structure_graph(clean_substrate)
            # Read all the edges, that are between adsorbate and surface (dist<2 condition)
            ads_edges = [(u, v, d) for u, v, d in full.edges.data() if d["dist"] < self.DIS_SURF2SURF]
            # take all the nodes that have an adsorbate atoms
            ads_nodes = [(n, d) for n, d in full.nodes.data() if d["index"] in input_ads_indices]
            # for (n, d) in ads_nodes:
            #    print(n, d)
            full = nx.Graph(clean_graph)
            full.add_nodes_from(ads_nodes)
            full.add_edges_from(ads_edges)
        else:
            ...

        return full

    def _generate_structure_graph(self, atoms_, ads_indices: List[int]=None):
        """ generate molecular graph for reaction detection
            this is part of process_atoms
        """
        # NOTE: create a new graph when this method is called
        graph = nx.Graph() 

        atoms = atoms_

        # init few params
        grid = self.pbc_grid

        # - add all atoms to graph
        natoms = len(atoms)
        for i in range(natoms):
            for x, y, z in grid_iterator(grid):
                self.add_atoms_node(graph, atoms[i].symbol, i, (x, y, z))   

        # - create a neighbour list
        nl = self.neigh_creator.build_neighlist(atoms, bothways=False)
        nl.update(atoms)

        # - add all edges to graph
        for centre_idx in range(natoms):
            for x, y, z in grid_iterator(grid):
                nei_indices, offsets = nl.get_neighbors(centre_idx)
                for nei_idx, offset in zip(nei_indices, offsets):
                    ox, oy, oz = offset
                    if not (-grid[0] <= ox + x <= grid[0]):
                        continue
                    if not (-grid[1] <= oy + y <= grid[1]):
                        continue
                    if not (-grid[2] <= oz + z <= grid[2]):
                        continue
                    # TODO: use tag to manually set dist between atoms in one adsorbate
                    # This line ensures that only surface-adsorbate bonds are accounted for that are less than 2.5 Å
                    dis = atoms.get_distances(centre_idx, nei_idx, mic=True)
                    if dis > self.substrate_adsorbate_distance and (bool(centre_idx in ads_indices) ^ bool(nei_idx in ads_indices)):
                        continue
                    centre_sym, nei_sym = atoms[centre_idx].symbol, atoms[nei_idx].symbol
                    self.add_atoms_edge(
                        graph, centre_sym, nei_sym, centre_idx, nei_idx, (x, y, z), (x + ox, y + oy, z + oz), 
                        atoms.get_distance(centre_idx, nei_idx, mic=True),
                        ads_indices
                    )

        return graph

def extract_chem_envs(
    graph: nx.Graph, atoms: Atoms, ads_indices: Optional[List[int]],
    graph_radius: int=2,
):
    """Extract chemical environments of selected species.
    
    Part of process_atoms. The species could either a single atom or a molecule.
    
    Args:
        graph: Atoms graph.
        atoms: Atoms object.
        ads_indices: A List of indices.
        graph_radius: The graph scope that defines a local environment.

    Returns:
        Chemical environments of species.

    """
    # 
    full = graph

    # - Get adsorbate graphs (single atom or molecule)
    # All adsorbates into single graph, no surface
    ads_nodes = None
    ads_nodes = [node_symbol(atoms[i].symbol, i, (0, 0, 0)) for i in ads_indices]
    ads_graphs = nx.subgraph(full, ads_nodes)

    # - get subgraphs, one for each molecule
    #ads_graphs = nx.connected_component_subgraphs(ads_graphs) # removed in v2.4
    # this creates a list of separate adsorbate graphs
    ads_graphs = [ads_graphs.subgraph(c) for c in nx.connected_components(ads_graphs)]
    #print("number of adsorbate graphs: ", len(ads_graphs))

    chem_envs = []
    for idx, ads in enumerate(ads_graphs):
        #print(f"----- adsorbate {idx} -----")
        #plot_graph(ads, fig_name=f"ads-{idx}.png")
        #print("ads nodes: ", ads.nodes())
        initial = list(ads.nodes())[0] # the first node in the adsorbate
        full_ads = nx.ego_graph(full, initial, radius=0, distance="ads_only") # all nodes in this adsorbate, equal ads?
        #print("full ads: ", full_ads.nodes())

        new_ads = nx.ego_graph(
            full, initial, 
            radius=(graph_radius*DIS_SURF2SURF)+DIS_ADS2SURF, 
            distance="dist"
        ) # consider neighbour atoms
        new_ads = nx.Graph(nx.subgraph(full, list(new_ads.nodes()))) # return a new copy of graph

        # update attribute of this adsorbate
        for node in ads.nodes():
            new_ads.add_node(node, central_ads=True) # means the atom is in [0,0,0]

        # update attr of this and neighbour adsorbates
        for node in full_ads.nodes():
            new_ads.add_node(node, ads=True)
        #print("new ads: ", new_ads.nodes())
        
        #plot_graph(new_ads, fig_name=f"ads-{idx}.png")

        chem_envs.append(new_ads)
    #exit()
    
    # - unique and sort
    #chem_envs = self.unique_adsorbates(chem_envs)  
    ## sort chem_env by number of edges
    #chem_envs.sort(key=lambda x: len(x.edges()))

    return chem_envs.copy()


if __name__ == "__main__":
    ...
