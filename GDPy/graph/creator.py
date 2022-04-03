#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import combinations

import numpy as np
import scipy

from typing import List

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
plt.style.use("presentation")

from ase import Atoms
from ase.io import read, write
from ase.constraints import constrained_indices

from ase.neighborlist import natural_cutoffs, NeighborList

import networkx as nx

from GDPy.graph.sites import AdsSite
from GDPy.graph.utils import node_symbol, bond_symbol, unpack_node_name
from GDPy.graph.utils import bond_match, ads_match
from GDPy.graph.utils import show_nodes, show_edges, plot_graph

""" detection of oxygen vacancy
"""


def normalize(vector):
    return vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else vector * 0

def offset_position(atoms, neighbor, offset):
   return atoms[neighbor].position + np.dot(offset, atoms.get_cell())

def plane_normal(xyz):
    """Return the surface normal vector to a plane of best fit. THIS CODE IS BORROWED FROM CATKIT

    Parameters
    ----------
    xyz : ndarray (n, 3)
        3D points to fit plane to.

    Returns
    -------
    vec : ndarray (1, 3)
        Unit vector normal to the plane of best fit.
    """
    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(xyz.shape[0])]
    vec, _, _, _ = scipy.linalg.lstsq(A, xyz[:, 2])
    vec[2] = -1.0

    vec /= -np.linalg.norm(vec)

    return vec

def get_angle_cycle(a1,a2,a3):
    v1 = a1-a2
    v2 = a3-a2
    #print(v1,v2)
    nv1 = np.linalg.norm(v1)
    nv2 = np.linalg.norm(v2)
    if (nv1 <= 0).any() or (nv2 <= 0).any():
        raise ZeroDivisionError('Undefined angle')
    v1 /= nv1
    v2 /= nv2
    #print(v1,v2)
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

class NeighGraphCreator():

    # ase nl params
    covalent_ratio = 1.0
    skin = 0.0 # TODO: the neighbours will be in cutoff + skin
    self_interaction = False
    bothways = True

    # nx-params
    graph_radius = 2 # graph radius 
    # help='Sets the graph radius, this can be tuned for different behavior')

    pbc_grid = [2, 2, 0] # for surface system

    def __init__(self, **kwargs):
        """"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return

    # ----- graph -----


    @staticmethod
    def grid_iterator(grid):
        """Yield all of the coordinates in a 3D grid as tuples

        Args:
            grid (tuple[int] or int): The grid dimension(s) to
                                      iterate over (x or (x, y, z))

        Yields:
            tuple: (x, y, z) coordinates
        """
        if isinstance(grid, int): # Expand to 3D grid
            grid = (grid, grid, grid)

        for x in range(-grid[0], grid[0]+1):
            for y in range(-grid[1], grid[1]+1):
                for z in range(-grid[2], grid[2]+1):
                    yield (x, y, z)
    
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

        return nl


class StruGraphCreator(NeighGraphCreator):

    """
        node
        edge
    """

    # split system atoms into framework and adsorbate
    adsorbate_elements = []
    adsorbate_indices = None

    substrate_adsorbate_distance = 2.5

    atoms = None
    nl = None
    graph = None
    ads_indices = None

    surface_mask = None

    clean = None

    def __init__(self, **kwargs):
        # NOTE: half neigh is enough for creating structure graph
        super().__init__(**kwargs)

        return

    def add_atoms_node(self, graph, a1, o1, **kwargs):
        """"""
        graph.add_node(
            node_symbol(self.atoms[a1], o1), index=a1, central_ads=False, **kwargs
        )

        return

    def add_atoms_edge(
        self, 
        graph,
        a1, a2, o1, o2, 
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

        dist = 2 - (1 if a1 in adsorbate_atoms else 0) - (1 if a2 in adsorbate_atoms else 0)

        graph.add_edge(
            # node index
            node_symbol(self.atoms[a1], o1),
            node_symbol(self.atoms[a2], o2),
            # attributes - edge data
            bond = bond_symbol(self.atoms, a1, a2),
            index = "{}:{}".format(*sorted([a1, a2])),
            dist = dist,
            dist_edge = self.atoms.get_distance(a1, a2, mic=True),
            ads_only = 0 if (a1 in adsorbate_atoms and a2 in adsorbate_atoms) else 2,
            **kwargs
        )

        return

    def check_system(atoms):
        """ whether molecule or periodic
        """
        atoms.cell = 20.0*np.eye(3)
        atoms.pbc = True
        atoms.center()
        print(atoms)

        return
    
    def generate_graph(self, atoms):
        if self.graph is not None:
            #raise RuntimeError(f"StruGraphCreator already has a graph...")
            print(f"overwrite stored graph...")
        
        # TODO: fix this, too complicated
        input_atoms = atoms.copy()
        full = self.generate_structure_graph(atoms)
        input_ads_indices = self.ads_indices.copy()

        if self.clean is not None: # BUG!!! need an alignment operation
            print("use clean substrate plus adsorbates...")
            clean_graph = self.generate_structure_graph(read(self.clean))
            ads_edges = [(u, v, d) for u, v, d in full.edges.data() if d["dist"] < 2]    ### Read all the edges, that are between adsorbate and surface (dist<2 condition)
            ads_nodes = [(n, d) for n, d in full.nodes.data() if d["index"] in input_ads_indices]    ### take all the nodes that have an adsorbate atoms
            #for (n, d) in ads_nodes:
            #    print(n, d)
            full = nx.Graph(clean_graph)
            full.add_nodes_from(ads_nodes)
            full.add_edges_from(ads_edges)
        #show_nodes(full)
        
        self.atoms = input_atoms
        self.ads_indices = input_ads_indices
        self.graph = full

        return

    def generate_structure_graph(self, atoms):
        """ generate molecular graph for reaction detection
            this is part of process_atoms
        """
        graph = nx.Graph() # NOTE: create a new graph when this method is called

        self.atoms = atoms
        self.nl = self.build_neighlist(self.atoms, bothways=False)
        self.nl.update(self.atoms)

        if self.adsorbate_indices is not None:
            self.ads_indices = self.adsorbate_indices.copy()
        else:
            self.ads_indices = [a.index for a in self.atoms if a.symbol in self.adsorbate_elements]
        print("adsorbates: ", self.ads_indices)

        # init few params
        grid = self.pbc_grid

        # Add all atoms to graph
        natoms = len(self.atoms)
        for i in range(natoms):
            for x, y, z in self.grid_iterator(grid):
                self.add_atoms_node(graph, i, (x, y, z))   
    
        # Add all edges to graph
        for centre_idx in range(natoms):
            for x, y, z in self.grid_iterator(grid):
                nei_indices, offsets = self.nl.get_neighbors(centre_idx)
                for nei_idx, offset in zip(nei_indices, offsets):
                    ox, oy, oz = offset
                    if not (-grid[0] <= ox + x <= grid[0]):
                        continue
                    if not (-grid[1] <= oy + y <= grid[1]):
                        continue
                    if not (-grid[2] <= oz + z <= grid[2]):
                        continue
                    # This line ensures that only surface adsorbate bonds are accounted for that are less than 2.5 Å
                    dis = atoms.get_distances(centre_idx, nei_idx, mic=True)
                    if dis > self.substrate_adsorbate_distance and (bool(centre_idx in self.ads_indices) ^ bool(nei_idx in self.ads_indices)):
                        continue
                    self.add_atoms_edge(graph, centre_idx, nei_idx, (x, y, z), (x + ox, y + oy, z + oz), self.ads_indices)
        
        return graph


    def extract_chem_envs(self):
        """ part of process_atoms
            return chemical environments of adsorbates
        """
        if self.graph is None:
            pass
            raise RuntimeError(f"{self.__name__} does not have a graph...")
        else:
            pass

        full = self.graph
    
        # All adsorbates into single graph, no surface
        ads_nodes = None
        ads_nodes = [node_symbol(self.atoms[index], (0, 0, 0)) for index in self.ads_indices]
        ads_graphs = nx.subgraph(full, ads_nodes)

        # Cut apart graph
        #ads_graphs = nx.connected_component_subgraphs(ads_graphs) # removed in v2.4
        # this creates a list of separate adsorbate graphs
        ads_graphs = [ads_graphs.subgraph(c) for c in nx.connected_components(ads_graphs)]
        #print("number of adsorbate graphs: ", len(ads_graphs))

        chem_envs = []
        for idx, ads in enumerate(ads_graphs):
            #print(f"----- adsorbate {idx} -----")
            #print("ads nodes: ", ads.nodes())
            initial = list(ads.nodes())[0] # the first node in the adsorbate
            full_ads = nx.ego_graph(full, initial, radius=0, distance="ads_only") # all nodes in this adsorbate, equal ads?
            #print("full ads: ", full_ads.nodes())

            new_ads = nx.ego_graph(full, initial, radius=(self.graph_radius*2)+1, distance="dist") # consider neighbour atoms
            new_ads = nx.Graph(nx.subgraph(full, list(new_ads.nodes()))) # return a new copy of graph

            # update attribute of this adsorbate
            for node in ads.nodes():
                new_ads.add_node(node, central_ads=True)

            # update attr of this and neighbour adsorbates
            for node in full_ads.nodes():
                new_ads.add_node(node, ads=True)
            #print("new ads: ", new_ads.nodes())
            
            #self.plot_graph(new_ads, fig_name=f"ads-{idx}.png")

            chem_envs.append(new_ads)

        chem_envs = self.unique_adsorbates(chem_envs)  

        # sort chem_env by number of edges
        chem_envs.sort(key=lambda x: len(x.edges()))

        return chem_envs.copy()

    def unique_adsorbates(self, chem_envs):
        """Removes duplicate adsorbates which occur when periodic
        boundary conditions detect the same adsorbate in two places. Each
        adsorbate graph has its atomic index checked to detect when PBC has 
        created duplicates.

        Args:
            chem_env_groups (list[networkx.Graph]): Adsorbate/environment graphs

        Returns:
            list[networkx.Graph]: The unique adsorbate graphs
        """
        # Keep track of known unique adsorbate graphs
        unique = []
        for env in chem_envs:
            for unique_env in unique:
                if nx.algorithms.isomorphism.is_isomorphic(
                    env, unique_env, edge_match=bond_match, node_match=ads_match
                ):
                    break
            else: # Was unique
                unique.append(env)

        return unique



class SiteGraphCreator(StruGraphCreator):

    """ procedure
        1. detect surface atoms
        2. find site with different orders (coordination number)
    """

    coordination_numbers = []

    no_adsorb = []

    surface_normal = 0.65

    site_radius = 2

    shift2 = 0.0
    shift3 = 0.0 # -0.7

    # bind an atoms object
    atoms = None
    nl = None
    graph = None
    ads_indices = None
    surface_mask = None

    clean = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        return
    
    def convert_atoms(self, atoms, check_unique=False):
        """"""
        if isinstance(self.atoms, Atoms):
            #raise RuntimeError("SiteGraphCreator already has an atoms object.")
            print("SiteGraphCreator already has an atoms object.")

        self.atoms = atoms

        # create graph
        _ = self.generate_graph(self.atoms) # create self.graph

        # NOTE: update nl since we want use nl with bothways instead of false
        self.nl = self.build_neighlist(self.atoms, bothways=True)
        self.nl.update(self.atoms)

        ### make sure to manually set the normals for 2-D materials, all atoms should have a normal pointing up, as all atoms are surface atoms
        normals, surface_mask = self.generate_normals(
            surface_normal=self.surface_normal, ads_indices=self.ads_indices, normalize_final=True
        )

        # remove constrained atoms from surface_mask
        constrained = constrained_indices(self.atoms)
        surface_mask = [index for index in surface_mask if index not in constrained]

        # remove unallowed elements
        surface_mask = [index for index in surface_mask if self.atoms[index].symbol not in self.no_adsorb]

        print("surface mask: ", surface_mask)

        self.atoms.arrays["surface_direction"] = normals

        # write("xxx.xyz", atoms)

        # check coord format
        # TODO: move to init?
        if isinstance(self.coordination_numbers, list):
            cn_values = self.coordination_numbers.copy()
            cn_atom_indices = [surface_mask.copy() for x in range(len(cn_values))]
        elif isinstance(self.coordination_numbers, dict):
            cn_values, cn_atom_indices = [], []
            for a, b in self.coordination_numbers.items():
                cn_values.append(int(a.split("_")[-1]))
                cn_atom_indices.append(b)
            print("cn_values: ", cn_values)
            print("cn_atom_indices: ", cn_atom_indices)
        else:
            raise RuntimeError("invalid coordination_number...")

        all_sites = []
        for coord, cur_surf_mask in zip(cn_values, cn_atom_indices):
            print("coordination number: ", coord)
            found_count = 0
            print("graph: ", self.graph)
            found_sites = self.generate_site_type(cur_surf_mask, surface_mask, normals, coordination=coord)

            # generate site graph
            for s in found_sites:
                s.graph = self.process_site(s.cycle, self.site_radius)

            # output sites
            #print("\n\nfound sites: ", len(found_sites))
            #for s in found_sites:
            #    print(s)

            if check_unique:
                site_envs = [None] * len(found_sites)
                for i, site in enumerate(found_sites):
                    site_envs[i] = [site.graph]
                    # print("new_site: ", new_site)

                unique_envs, unique_sites = unique_chem_envs(site_envs, found_sites)
                new_unique_sites = [None] * len(unique_sites)
                for i, site_group in enumerate(unique_sites):
                    xposes = [s.position[0] for s in site_group]
                    yposes = [s.position[1] for s in site_group]
                    zposes = [s.position[2] for s in site_group]
                    sorted_indices = np.lexsort((xposes,yposes,zposes))
                    new_unique_sites[i] = [site_group[s_idx] for s_idx in sorted_indices] 
                unique_sites = new_unique_sites
            else:
                unique_sites = [[s] for s in found_sites]

            #print("\n\nunique sites: ", len(unique_sites))
            #for s in unique_sites:
            #    print(s)
            all_sites.extend(unique_sites)
        
        # unique again
        if len(cn_values) > 1:
            found_sites = [s[0] for s in all_sites]
            if check_unique:
                site_envs = [None] * len(found_sites)
                for i, site in enumerate(found_sites):
                    site_envs[i] = [site.graph]
                    # print("new_site: ", new_site)

                unique_envs, unique_sites = unique_chem_envs(site_envs, found_sites)
                new_unique_sites = [None] * len(unique_sites)
                for i, site_group in enumerate(unique_sites):
                    xposes = [s.position[0] for s in site_group]
                    yposes = [s.position[1] for s in site_group]
                    zposes = [s.position[2] for s in site_group]
                    sorted_indices = np.lexsort((xposes,yposes,zposes))
                    new_unique_sites[i] = [site_group[s_idx] for s_idx in sorted_indices] 
                unique_sites = new_unique_sites
            else:
                unique_sites = [[s] for s in found_sites]
            all_sites = unique_sites

        return all_sites

 
    def generate_site_type(
        self, 
        valid_surface_mask,
        surface_mask, # indices of surface atoms that can form sites
        normals, 
        coordination
    ) -> List[AdsSite]:
        """"""
        # NOTE: check if is None
        atoms = self.atoms 
        nl = self.nl

        possible = list(combinations(set(valid_surface_mask), coordination))

        # check if the pair is neighbour
        valid = []
        for cycle in possible:
           for start, end in combinations(cycle, 2):
               if end not in self.nl.get_neighbors(start)[0]:
                   break
           else: # All were valid
                valid.append(list(cycle))

        #print(valid)
        sites = []
        for cycle in valid:
            #if cycle !=  [32, 33, 27]:
            #    continue
            #print("\n\n --- cycle --- ", cycle)
            
            # ***** found local site structure *****
            site_node_names = []
            for u, d in self.graph.nodes.data():
                #print(u, d)
                if d["index"] in cycle:
                    site_node_names.append(u)
            site_graph = nx.subgraph(self.graph, site_node_names)
            #site_graphs = [site_graph.subgraph(c) for c in nx.connected_components(site_graph)]
            #for i, x in enumerate(site_graphs):
            #    print(f"--- {i} ---")
            #    print(x)
            #plot_graph(site_graph, "sg.png")
            centre_node = node_symbol(self.atoms[cycle[0]], (0,0,0))
            #print("centre: ", centre_node)
            local_graph = nx.ego_graph(
                site_graph, centre_node, 
                radius=2, center=True,
                distance="dist"
            ) # go over framework
            #plot_graph(local_graph, "local.png")

            site_node_names = []
            if coordination == 3:
                #print("node names: ", local_graph.nodes())
                all_cliques = nx.enumerate_all_cliques(local_graph)
                site_node_names = [x for x in all_cliques if len(x)==3]
            # TODO: add CN4 site that is a square
            else:
                conn_nodes = []
                for u, d in local_graph.degree:
                    if d == coordination-1:
                        conn_nodes.append(u)
                new_site_graph = nx.subgraph(self.graph, conn_nodes)
                #plot_graph(new_site_graph, "sg-%s.png" %("-".join(str(c) for c in cycle)))
                new_site_graphs = [new_site_graph.subgraph(c) for c in nx.connected_components(new_site_graph)]

                # *** check if node number is valid
                if len(new_site_graphs) == 1:
                    if len(new_site_graphs[0]) == coordination:
                        site_node_names = [list(new_site_graphs[0]).copy()]
                else:
                    for g in new_site_graphs:
                        if len(g) == coordination:
                            names = list(g).copy()
                            names.append(centre_node)
                            site_node_names.append(names)

            for names in site_node_names:
                #print("xxx")
                #print(names)
                # calc centre
                site_positions = np.zeros((coordination,3))
                for ci, n in enumerate(names):
                    chem_sym, a_idx, offset = unpack_node_name(n)
                    site_positions[ci] = self.atoms.positions[a_idx] + np.dot(offset, self.atoms.get_cell())
                    #print(cycle[ci], site_positions[ci], offset)
                average = np.average(site_positions, axis=0)
                #print("site pos: ", site_positions)

                # generate site position
                """
                # TODO: is it okay for order 4 site?
                tracked = np.array(atoms[cycle[0]].position, dtype=float)
                known = np.zeros(shape=(coordination, 3), dtype=float)
                known[0] = tracked
                print(known)
                for index, (start, end) in enumerate(zip(cycle[:-1], cycle[1:])):
                    nei_indices, offsets = nl.get_neighbors(start)
                    print("start: ", start, end, nei_indices)
                    for neighbor, offset in zip(nei_indices, offsets):
                        if neighbor == end:
                            tracked += offset_position(atoms, neighbor, offset) - atoms[start].position
                            known[index + 1] = tracked

                average = np.average(known, axis=0)
                """

                if coordination ==2:
                    average[2] = average[2] + self.shift2
                if coordination == 3:
                    average[2] = average[2] + self.shift3 

                # --- generate site normal --- TODO: ?
                normal = np.zeros(3)

                if len(cycle) == 1:
                    for index in cycle:
                        neighbors = len(nl.get_neighbors(index)[0])
                        normal += normals[index] * (1/neighbors)
                    normal = normalize(normal)

                if len(cycle) > 1: # TODO?
                    index = len(cycle) - 1
                    neighbors = len(nl.get_neighbors(index)[0])
                    cycle_orig = cycle
                    #print(cycle)
                    normal = self.generate_normals_new(atoms, cycle_orig, surface_mask)
                    #print(cycle,normal)
                for index in cycle:
                    neighbors = len(nl.get_neighbors(index)[0])
                    normal += normals[index] * (1/neighbors)
                normal = normalize(normal)

                #print("end cycle: ", cycle)
                #print("final normal: ", normal)
                #print("site pos: ", average)
                
                # --- create site objetc ---
                site_ads = AdsSite(
                    atoms=self.atoms.copy(), 
                    cycle=cycle, 
                    position=average, normal=normal
                )
                sites.append(site_ads)

        return sites
    
    def process_site(
        self, 
        site, # cycle
        radius=3
    ):
        """ steps
            1. add a placeholder X node
            2. find all neighbours of site atoms
            this can include built-in adsorbates
        """
        #print("radius: ", radius)
        full = self.graph
        #neighbors = nl.neighbors
        #offsets = nl.displacements
        #neighbors, offsets = nl.get_neighbors()
        full.add_node("X", index=None)
        offset = np.array([0, 0, 0])
        full.add_edge(
            "X",
            node_symbol(self.atoms[site[0]], offset),
            bond="X:{}".format(self.atoms[site[0]].symbol),
            ads=0
        ) # Handle first manually

        for last_index, next_index in zip(site[:-1], site[1:]):
            # Error handling needed, .index could be None / -1?
            neighbors, offsets = self.nl.get_neighbors(last_index)
            #neighbor_index = list(neighbors[last_index]).index(next_index)
            neighbor_index = list(neighbors).index(next_index)
            #offset += offsets[last_index][neighbor_index]
            offset += offsets[neighbor_index]
            #print(offset)
            full.add_edge(
                "X",
                node_symbol(self.atoms[next_index], offset),
                bond="X:{}".format(self.atoms[next_index].symbol),
                ads=0
            )

        # TODO: ads also in site_graph?
        site_graph = nx.ego_graph(full, "X", radius=(radius*2)+1, distance="dist")
        site_graph = nx.subgraph(full, list(site_graph.nodes()))
        site_graph = nx.Graph(site_graph)
        full.remove_node("X")

        return site_graph
    
    def generate_normals(
        self,
        surface_normal=0.5, normalize_final=True, ads_indices=[]
    ):
        """"""
        # NOTE: original code has bug
        atoms = self.atoms.copy()
        #del atoms[adsorbate_atoms]

        normals = np.zeros(shape=(len(atoms), 3), dtype=float)

        for index, atom in enumerate(atoms):
            # print("centre: ", index, atom.position)
            if index in ads_indices:
                continue
            normal = np.array([0, 0, 0], dtype=float)
            for neighbor, offset in zip(*self.nl.get_neighbors(index)):
                if neighbor in ads_indices:
                    continue
                # print("neigh: ", neighbor)
                direction = atom.position - offset_position(atoms, neighbor, offset)
                normal += direction
            # print("direction:", normal)
            if np.linalg.norm(normal) > surface_normal:
                normals[index,:] = normalize(normal) if normalize_final else normal

        # NOTE: assign z direction for surface
        #surface_mask = [index for index in range(len(atoms)) if np.linalg.norm(normals[index]) > 1e-5]
        surface_mask = [index for index in range(len(atoms)) if np.linalg.norm(normals[index]) > 1e-5 and normals[index][2] > 0.]
        if self.surface_mask is not None:
            print("use input surface mask...")
            surface_mask = self.surface_mask
            for i in range(len(atoms)):
                if i not in surface_mask:
                    normals[i] = np.zeros(3)

        return normals, surface_mask

    def generate_normals_new(self, atoms, cycle, surface_mask):
        """"""
        site_atoms = cycle.copy()
        if len(site_atoms) > 2:
            atom_array = []
            for a in site_atoms:
                atom_array.append(atoms[a].position)
            normal = plane_normal(np.array(atom_array)) # TODO?
        else: # for bridge site
            neighbor_atoms = []
            #print(self.nl.get_neighbors(site_atoms[0])[0])
            #print(self.nl.get_neighbors(site_atoms[1])[0])
            for i in self.nl.get_neighbors(site_atoms[0])[0]:
                #NOTE: This if condition is to ensure that if the atoms are the same row, 
                # then the plane is formed btwn an atom in another row
                # print(i,atoms[i].position[2] - atoms[site_atoms[0]].position[2])
                if (i not in site_atoms) and i in self.nl.get_neighbors(site_atoms[1])[0] and i in surface_mask and i not in neighbor_atoms:
                    neighbor_atoms.append(i)
            #print("neighs: ", neighbor_atoms)

            normal = [0, 0, 0]
            if len(neighbor_atoms) > 0:
                for i in neighbor_atoms:
                    site_atoms1 = site_atoms.copy()
                    site_atoms1.append(i)
                    atom_array = []
                    initial = site_atoms[0]
                    atom_array.append(atoms[initial].position)
                    for a in site_atoms1:
                        if a != initial:
                            a_offset = self.nl.get_neighbors(initial)[1][np.where(a==self.nl.get_neighbors(initial)[0])]
                            #print(a,np.dot(a_offset, atoms.get_cell())+atoms[a].position)
                            atom_array.append(atoms[a].position+np.dot(a_offset, atoms.get_cell())[0])
                    #print(atom_array)
                    #print(get_angle_cycle(*atom_array))
                    # NOTE: angle settings
                    if 0.85 < get_angle_cycle(atom_array[0],atom_array[1],atom_array[2]) < 1.2: # 48.73 to 68.79 degree
                        normal += plane_normal(np.array(atom_array))
                    #normal += plane_normal(np.array(atom_array))
                        #print('using this cycle to add to normal')
                    #print(normal)
            # check normal
            if np.sum(np.fabs(normal)) < 1e-8:
                initial = site_atoms[0]
                atom_array = []
                atom_array.append(atoms[initial].position)
                for a in site_atoms:
                    if a != initial:
                        a_offset = self.nl.get_neighbors(initial)[1][np.where(a==self.nl.get_neighbors(initial)[0])]
                        #print(a,np.dot(a_offset, atoms.get_cell())+atoms[a].position)
                        atom_array.append(atoms[a].position+np.dot(a_offset, atoms.get_cell())[0])
                vec = atom_array[1] - atom_array[0]
                nvec = vec / np.linalg.norm(vec)
                normal = nvec + np.array([0.,0.,2**0.5])

            #print("func normal: ", normal)
        normal = normalize(normal)

        #print("func normal: ", normal)

        return normal

    # choose best sites to add adsorbate
    def check_valid_adsorb_site(self, unique_sites):
        """ choose a best site to adsorb in each unique site group
        """
        center = self.atoms.get_center_of_mass()
        for index, sites in enumerate(unique_sites):
            new = atoms.copy()
            best_site = sites[0]

            # NOTE: choose site is nearest to the COM
            for site in sites[1:]:
                if np.linalg.norm(site.position - center) < np.linalg.norm(best_site.position - center):
                    best_site = site
            #print(best_site.adsorb(new, ads, adsorbate_atoms),args.min_dist)
            ### this check is to ensure, that sites really close are not populated
            if best_site.adsorb(new, ads, adsorbate_atoms) >= args.min_dist:
                found_count += 1
                ### if hydrogen bonds exist then use this loop to populate these structures
                #H_bond_movie = orient_H_bond(new)
                H_bond_movie = []
                #print(H_bond_movie[:])
                if len(H_bond_movie) > 0:
                    for i in H_bond_movie:
                        movie.append(i)
                else:
                    movie.append(new)
                all_unique.append(site)
        
        return
    


if __name__ == "__main__":
    pass
