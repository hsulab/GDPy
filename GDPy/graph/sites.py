#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

from typing import List

from itertools import combinations

import numpy as np
import scipy as sp
import networkx as nx

from ase import Atoms
from ase.constraints import constrained_indices
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.graph.creator import StruGraphCreator
from GDPy.graph.comparison import bond_match, get_unique_environments_based_on_bonds
from GDPy.graph.utils import node_symbol, unpack_node_name, show_nodes


def make_clean_atoms(atoms_, results=None):
    """Create a clean atoms from the input."""
    atoms = Atoms(
        symbols=atoms_.get_chemical_symbols(),
        positions=atoms_.get_positions().copy(),
        cell=atoms_.get_cell().copy(),
        pbc=copy.deepcopy(atoms_.get_pbc())
    )
    if results is not None:
        spc = SinglePointCalculator(atoms, **results)
        atoms.calc = spc

    return atoms

def get_angle_cycle(a1,a2,a3):
    """Get the angle by three vectors."""
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
    vec, _, _, _ = sp.linalg.lstsq(A, xyz[:, 2])
    vec[2] = -1.0

    vec /= -np.linalg.norm(vec)

    return vec

def normalize(vector):
    return vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else vector * 0

class SingleAdsorptionSite(object):

    MIN_INTERADSORBATE_DISTANCE = 1.5

    """A site for single-dentate adsorption.
    """

    def __init__(
        self, atoms: Atoms, nl, normals,
        site_indices: List[int], node_names: List[str], surf_indices: List[int],
        graph=None, *args, **kwargs
    ):
        """

        Args:
            site_indices: Indices of site atoms.

        """
        self.atoms = make_clean_atoms(atoms)

        #print(offsets)
        #assert len(cycle) == len(offsets), "site atomic number is inconsistent..."
        self.site_indices = site_indices
        #self.offsets = offsets
        #self.known = known

        order = len(self.site_indices)
        
        self.position, self.normal = self._compute_position(
            atoms, nl, normals, node_names, order, site_indices, surf_indices
        )

        self.graph = graph

        return

    def __eq__(self, other):
        return nx.is_isomorphic(self.graph, other.graph, edge_match=bond_match)

    def __repr__(self):
        content = (
            "Cycle:{}, " + 
            "Position: [" + "{:<8.4f}, "*3 +"], " + 
            "Normal: [" + "{:8.4f}, "*3 + "], " +
            "Graph: {}"
        ).format(self.site_indices, *self.position, *self.normal, bool(self.graph))
        return content

    def _compute_position(
        self, atoms: Atoms, nl, normals, node_names: List[str], order: int,
        site_indices: List[int], surf_indices: List[int]
    ):
        """Compute the position of the site based on the structure graph.

        Args:
            atoms: Structure.
            order: Coordination numebr of the site (top 1, bridge 2, ...).
        
        Returns:
            Site positions and its plane normal.

        """
        assert len(site_indices) == order, "Site order is inconsistent."
        
        # - find positions of atoms forming the site and the normal
        #print("xxx")
        #print(names)
        # calc centre
        site_positions = np.zeros((order,3))
        for ci, n in enumerate(node_names):
            chem_sym, a_idx, offset = unpack_node_name(n)
            site_positions[ci] = atoms.positions[a_idx] + np.dot(offset, atoms.get_cell())
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

        # --- NOTE: add default distance to the site
        #if coordination == 2:
        #    average[2] = average[2] + self.shift2
        #if coordination == 3:
        #    average[2] = average[2] + self.shift3 

        # --- generate site normal --- TODO: ?
        normal = np.zeros(3)

        if order == 1:
            for index in site_indices:
                neighbors = len(nl.get_neighbors(index)[0])
                normal += normals[index] * (1/neighbors)
            normal = normalize(normal)
        else:
            if order > 1: # TODO?
                index = order - 1
                neighbors = len(nl.get_neighbors(index)[0])
                cycle_orig = order
                #print(cycle)
                normal = compute_site_normal(atoms, nl, site_indices, surf_indices)
                #print(cycle,normal)
            else:
                # NOTE: order < 1 is not allowed
                ...

        for index in site_indices:
            neighbors = len(nl.get_neighbors(index)[0])
            normal += normals[index] * (1/neighbors)
        normal = normalize(normal)

        return average, normal
    
    def is_occupied(self, ads_elemnts):
        """"""
        ads_indices = [a.index for a in self.atoms if a.symbol in ads_elemnts]
        ads_nodes = None
        ads_nodes = [node_symbol(self.atoms[i], (0, 0, 0)) for i in ads_indices]

        ads_graphs = nx.subgraph(self.graph, ads_nodes)
        ads_graphs = [ads_graphs.subgraph(c) for c in nx.connected_components(ads_graphs)]
        print("number of adsorbate graphs: ", len(ads_graphs))
        #print(self.offsets)

        for idx, ads in enumerate(ads_graphs):
            print(f"----- adsorbate {idx} -----")
            print("ads nodes: ", ads.nodes())
            initial = list(ads.nodes())[0] # the first node in the adsorbate
            full_ads = nx.ego_graph(self.graph, initial, radius=0, distance="ads_only") # all nodes in this adsorbate, equal ads?
            #print("full ads: ", full_ads.nodes())

            new_ads = nx.ego_graph(self.graph, initial, radius=1, distance="dist") # consider neighbour atoms
            new_ads = nx.Graph(nx.subgraph(self.graph, list(new_ads.nodes()))) # return a new copy of graph

            # update attribute of this adsorbate
            for node in ads.nodes():
                new_ads.add_node(node, central_ads=True) # central means the adsorbate interacting with surface

            # update attr of this and neighbour adsorbates
            for node in full_ads.nodes():
                new_ads.add_node(node, ads=True)

            show_nodes(new_ads)

        return

    def adsorb(
        self, 
        adsorbate, # single atom or a molecule
        other_ads_indices, # indices of other adsorbate
        distance_to_site=1.5, check_H_bond=False
    ):
        """"""
        # --- prepare and add adsorbate
        atoms = self.atoms.copy()
        ads_copy = adsorbate.copy()
        ads_copy.rotate([0, 0, 1], self.normal, center=[0,0,0])
        #print(self.position,self.position+ (self.normal*height), self.normal)
        ads_copy.translate(self.position + (self.normal*distance_to_site))

        atoms.extend(ads_copy)

        # --- check indices of adsorbed species
        index_to_check = range(len(atoms)-len(ads_copy), len(atoms))
        index_to_check_noH = [] # indices of to-put adsorbate
        for ads_t in index_to_check:
            if atoms[ads_t].symbol != 'H':
                index_to_check_noH.append(ads_t)

        ads_atoms_check = [] # indices of other adsorbates
        for ads_t in other_ads_indices:
            if atoms[ads_t].symbol != 'H':
                ads_atoms_check.append(ads_t)
        #print(index_to_check, index_to_check_noH)

        # --- check distances between adsorbates
        all_ads_indices = other_ads_indices.copy()
        for ad in range(len(atoms)-len(ads_copy), len(atoms)):
            all_ads_indices.append(ad)
        #print(ads_atoms)

        dist = float("inf")
        if len(other_ads_indices) != 0:
            for index in index_to_check_noH:
                dists = atoms.get_distances(index, ads_atoms_check, mic=True)
                dist = min(dist, dists.min())
            # TODO: check is_occupied
            if dist < self.MIN_INTERADSORBATE_DISTANCE:
                atoms = dist
        
        return atoms 


def generate_normals(
    atoms: Atoms, nl, surf_indices: List[int]=None, ads_indices: List[int]=[], 
    mask_elements: List[str]=[],
    surf_norm_min: float=0.5, normalised=True
):
    """Find normals to surface of a structure.

    Atoms that have positive z-axis normal would be on the surface.

    Args:
        atoms: Atoms object.
        nl: Pre-created neighbour list.
        surf_indices: Indices of surface atoms.
        ads_indices: Indices of adsorbate atoms.
        mask_elements: Indices of atoms that are not considered for adsorption sites.
        surf_norm_min: Minimum for a surface normal.
        normalised: Whether normalise normals.

    """
    natoms = len(atoms)

    normals = np.zeros(shape=(natoms, 3), dtype=float)
    for i, atom in enumerate(atoms):
        # print("centre: ", index, atom.position)
        if i in ads_indices:
            continue
        normal = np.array([0, 0, 0], dtype=float)
        for nei_idx, offset in zip(*nl.get_neighbors(i)):
            if nei_idx in ads_indices:
                continue
            # print("neigh: ", neighbor)
            direction = atom.position - (atoms[nei_idx].position + np.dot(offset, atoms.get_cell()))
            normal += direction
        # print("direction:", normal)
        if np.linalg.norm(normal) > surf_norm_min:
            normals[i,:] = normalize(normal) if normalised else normal

    # NOTE: check normal is pointed to z-axis surface
    if surf_indices is not None:
        #print("use input surface mask...")
        surf_indices_ = surf_indices
        for i in range(natoms):
            if i not in surf_indices_:
                normals[i] = np.zeros(3)
    else:
        #surface_mask = [index for index in range(len(atoms)) if np.linalg.norm(normals[index]) > 1e-5]
        surf_indices_ = [
            i for i in range(natoms) if np.linalg.norm(normals[i]) > 1e-5 and normals[i][2] > 0.
        ]

    # - remove constrained atoms from surface atoms
    constrained = constrained_indices(atoms)
    surf_indices_ = [i for i in surf_indices_ if i not in constrained]

    # - remove unallowed elements
    surf_indices_ = [i for i in surf_indices_ if atoms[i].symbol not in mask_elements]

    return normals, surf_indices_

def compute_site_normal(atoms: Atoms, nl, site_indices: List[int], surf_indices: List[int]):
    """Compute the plane normal of the site.

    Atoms that have positive z-axis normal would be on the surface.

    Args:
        atoms: Atoms object.
        nl: Pre-created neighbour list.
        site_indices: Indices of atoms that form the adsorption site.
        surf_indices: Indices of surface atoms.
    
    Returns:
        An array with shape (3,).

    """
    site_indices = copy.deepcopy(site_indices)
    site_order = len(site_indices)
    if site_order > 2:
        #atom_array = []
        #for a in site_indices:
        #    atom_array.append(atoms[a].position)
        positions = atoms.positions[site_indices,:]
        normal = plane_normal(positions) # TODO?
    else: # for bridge site
        neighbor_atoms = []
        #print(self.nl.get_neighbors(site_atoms[0])[0])
        #print(self.nl.get_neighbors(site_atoms[1])[0])
        for i in nl.get_neighbors(site_indices[0])[0]:
            #NOTE: This if condition is to ensure that if the atoms are the same row, 
            # then the plane is formed btwn an atom in another row
            # print(i,atoms[i].position[2] - atoms[site_atoms[0]].position[2])
            if (i not in site_indices) and i in nl.get_neighbors(site_indices[1])[0] and i in surf_indices and i not in neighbor_atoms:
                neighbor_atoms.append(i)
        #print("neighs: ", neighbor_atoms)

        normal = [0, 0, 0]
        if len(neighbor_atoms) > 0:
            for i in neighbor_atoms:
                site_atoms1 = site_indices.copy()
                site_atoms1.append(i)
                atom_array = []
                initial = site_indices[0]
                atom_array.append(atoms[initial].position)
                for a in site_atoms1:
                    if a != initial:
                        a_offset = nl.get_neighbors(initial)[1][np.where(a==nl.get_neighbors(initial)[0])]
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
            initial = site_indices[0]
            atom_array = []
            atom_array.append(atoms[initial].position)
            for a in site_indices:
                if a != initial:
                    a_offset = nl.get_neighbors(initial)[1][np.where(a==nl.get_neighbors(initial)[0])]
                    #print(a,np.dot(a_offset, atoms.get_cell())+atoms[a].position)
                    atom_array.append(atoms[a].position+np.dot(a_offset, atoms.get_cell())[0])
            vec = atom_array[1] - atom_array[0]
            nvec = vec / np.linalg.norm(vec)
            normal = nvec + np.array([0.,0.,2**0.5])

        #print("func normal: ", normal)
    normal = normalize(normal)

    #print("func normal: ", normal)

    return normal

def find_valid_indices(atoms, species: None, spec_indices=None, region=None):
    """Find indices of atoms based on criteria."""
    chemical_symbols = atoms.get_chemical_symbols()

    if spec_indices is None:
        if region is None:
            valid_indices = [] # selected indices
            for i, sym in enumerate(chemical_symbols):
                if sym == species:
                    valid_indices.append(i)
        else:
            valid_indices = []
            #print(region)
            (ox, oy, oz, xl, yl, zl, xh, yh, zh) = region
            for i, a in enumerate(atoms):
                if a.symbol == species:
                    pos = a.position
                    if (
                        (ox+xl <= pos[0] <= ox+xh) and
                        (oy+yl <= pos[1] <= oy+yh) and
                        (oz+zl <= pos[2] <= oz+zh)
                    ):
                        valid_indices.append(i)
    else:
        valid_indices = copy.deepcopy(spec_indices)
    
    return valid_indices


class SiteFinder(StruGraphCreator):

    """ procedure
        1. detect surface atoms
        2. find site with different orders (coordination number)
    """

    surface_normal = 0.65

    _site_radius = 2

    # bind an atoms object
    atoms = None
    nl = None
    graph = None

    def __init__(self, site_radius=2, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        # TODO: set params...
        #pbc_grid = kwargs.get("pbc_grid", [2,2,0])
        #graph_radius = kwargs.get("graph_radius", 2)
        #self.neigh_creator = NeighGraphCreator(
        #    covalent_ratio=1.0, skin=0.0, self_interaction=False, bothways=True, 
        #    graph_radius=graph_radius, pbc_grid=pbc_grid, *args, **kwargs
        #)
        self._site_radius = site_radius

        return
    
    @property
    def site_radius(self):

        return self._site_radius
    
    def find(self, atoms_: Atoms, site_params: list, surface_mask: List[int]=None, check_unique=False):
        """Find all sites based on graph.

        The site finding includes several steps. 1. Create a graph representation of 
        the structure. 2. Determine the surface normal. 3. Determine valid sites.

        Args:
            site_params: Parameters that define the site to find.

        """
        #if isinstance(self.atoms, Atoms):
        #    #raise RuntimeError("SiteGraphCreator already has an atoms object.")
        #    print("SiteGraphCreator already has an atoms object.")

        atoms = atoms_

        # - create graph
        _ = self.generate_graph(atoms) # create self.graph
        graph = self.graph

        # NOTE: update nl since we want use nl with bothways instead of false
        nl = self.neigh_creator.build_neighlist(atoms, bothways=True)
        nl.update(atoms)

        # - find atoms that are on surface and possibly form adsorption sites...
        # NOTE: make sure to manually set the normals for 2-D materials, 
        #       all atoms should have a normal pointing up, as all atoms are surface atoms
        #normals, surface_mask = self.generate_normals(
        #    surface_normal=self.surface_normal, ads_indices=self.ads_indices, normalize_final=True
        #)
        normals, surface_mask = generate_normals(
            atoms, nl, surface_mask, 
            self.ads_indices, mask_elements=[],
            surf_norm_min=self.surface_normal, normalised=True
        )
        print("surface mask: ", surface_mask)
        print("surface mask ads: ", self.ads_indices)

        atoms.arrays["surface_direction"] = normals
        # write("xxx.xyz", atoms)

        # - find all valid adsorption sites
        site_groups = []
        for cur_params in site_params:
            # -- some basic params
            cn = cur_params.get("cn") # coordination number
            cur_site_radius = cur_params.get("radius", self._site_radius)
            print("coordination number: ", cn)
            # -- find possible atoms to form the site
            print("graph: ", graph)
            cur_species = cur_params.get("species", None)
            site_indices = cur_params.get("site_indices", None)
            region = cur_params.get("region", None)
            cur_surf_mask = find_valid_indices(
                atoms, cur_species, spec_indices=site_indices, region=region
            )
            # -- create sites
            found_sites = self._generate_site(
                atoms, graph, nl,
                cur_surf_mask, surface_mask, normals, coordination=cn
            )

            # generate site graph
            for s in found_sites:
                s.graph = self.process_site(
                    atoms, graph, nl, s.site_indices, cur_site_radius
                )

            # -- get sites with unique environments
            #print("\n\nfound sites: ", len(found_sites))
            #for s in found_sites:
            #    print(s, s.graph)
            site_graphs = [s.graph for s in found_sites]
            unique_indices = get_unique_environments_based_on_bonds(site_graphs)
            #print(unique_indices)
            unique_sites = [found_sites[i] for i in unique_indices]

            #if check_unique:
            #    site_envs = [None] * len(found_sites)
            #    for i, site in enumerate(found_sites):
            #        site_envs[i] = [site.graph]
            #        # print("new_site: ", new_site)

            #    unique_envs, unique_sites = unique_chem_envs(site_envs, found_sites)
            #    new_unique_sites = [None] * len(unique_sites)
            #    for i, site_group in enumerate(unique_sites):
            #        xposes = [s.position[0] for s in site_group]
            #        yposes = [s.position[1] for s in site_group]
            #        zposes = [s.position[2] for s in site_group]
            #        sorted_indices = np.lexsort((xposes,yposes,zposes))
            #        new_unique_sites[i] = [site_group[s_idx] for s_idx in sorted_indices] 
            #    unique_sites = new_unique_sites
            #else:
            #    unique_sites = [[s] for s in found_sites]

            #print("\n\nunique sites: ", len(unique_sites))
            #for s in unique_sites:
            #    print(s)
            #all_sites.extend(unique_sites)
            site_groups.append(unique_sites)
        
        # unique again
        # TODO: why unique again?
        #if len(cn_values) > 1:
        #    found_sites = [s[0] for s in all_sites]
        #    if check_unique:
        #        site_envs = [None] * len(found_sites)
        #        for i, site in enumerate(found_sites):
        #            site_envs[i] = [site.graph]
        #            # print("new_site: ", new_site)

        #        unique_envs, unique_sites = unique_chem_envs(site_envs, found_sites)
        #        new_unique_sites = [None] * len(unique_sites)
        #        for i, site_group in enumerate(unique_sites):
        #            xposes = [s.position[0] for s in site_group]
        #            yposes = [s.position[1] for s in site_group]
        #            zposes = [s.position[2] for s in site_group]
        #            sorted_indices = np.lexsort((xposes,yposes,zposes))
        #            new_unique_sites[i] = [site_group[s_idx] for s_idx in sorted_indices] 
        #        unique_sites = new_unique_sites
        #    else:
        #        unique_sites = [[s] for s in found_sites]
        #    all_sites = unique_sites

        return site_groups
 
    def _generate_site(
        self, 
        atoms,
        graph,
        nl,
        valid_surface_mask,
        surface_mask, # indices of surface atoms that can form sites
        normals, 
        coordination: int
    ) -> List[SingleAdsorptionSite]:
        """Create adsorption sites.

        Args:
            coordination: Coordination number (order) of the site.
        """
        # NOTE: check if is None
        possible = list(combinations(set(valid_surface_mask), coordination))

        # check if the pair is neighbour
        valid = []
        for cycle in possible:
           for start, end in combinations(cycle, 2):
               if end not in nl.get_neighbors(start)[0]:
                   break
           else: # All were valid
                valid.append(list(cycle))

        #print(valid)
        sites = []
        for cycle in valid:
            # - find nodes form the site
            #   NOTE: the same atoms may form different sites due to PBC
            #         if the cell is very small...
            site_node_names = []
            for u, d in graph.nodes.data():
                #print(u, d)
                if d["index"] in cycle:
                    site_node_names.append(u)
            site_graph = nx.subgraph(graph, site_node_names)
            #site_graphs = [site_graph.subgraph(c) for c in nx.connected_components(site_graph)]
            #for i, x in enumerate(site_graphs):
            #    print(f"--- {i} ---")
            #    print(x)
            #plot_graph(site_graph, "sg.png")
            centre_node = node_symbol(atoms[cycle[0]].symbol, cycle[0], (0,0,0))
            #print("centre: ", centre_node)
            local_graph = nx.ego_graph(
                site_graph, centre_node, 
                radius=2, center=True, # 2 is surf_surf_dis
                distance="dist"
            ) # go over framework
            #plot_graph(local_graph, "local.png")

            site_node_names = [] # List[List[str]]
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
                new_site_graph = nx.subgraph(graph, conn_nodes)
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

            # --- create site object ---
            for node_names in site_node_names:
                site_ads = SingleAdsorptionSite(
                    atoms=atoms, nl=nl, normals=normals, site_indices=cycle,
                    node_names=node_names, surf_indices=surface_mask,
                )
                sites.append(site_ads)

        return sites
    
    def process_site(
        self, 
        atoms,
        graph,
        nl,
        site: List[int],
        radius=3
    ):
        """Create the graph representation of the site.

        Args:
            site: Indices of atoms that form the site.
        
        Steps:
            1. add a placeholder X node
            2. find all neighbours of site atoms
            this can include built-in adsorbates
        """
        # - add a placeholder X node in the box [0,0,0]
        #print("radius: ", radius)
        full = graph
        #neighbors = nl.neighbors
        #offsets = nl.displacements
        #neighbors, offsets = nl.get_neighbors()
        full.add_node("X", index=None)
        offset = np.array([0, 0, 0])
        full.add_edge(
            "X",
            node_symbol(atoms[site[0]].symbol, site[0], offset),
            bond="X:{}".format(atoms[site[0]].symbol),
            ads=0
        ) # Handle first manually

        for last_index, next_index in zip(site[:-1], site[1:]):
            # Error handling needed, .index could be None / -1?
            neighbors, offsets = nl.get_neighbors(last_index)
            #neighbor_index = list(neighbors[last_index]).index(next_index)
            neighbor_index = list(neighbors).index(next_index)
            #offset += offsets[last_index][neighbor_index]
            offset += offsets[neighbor_index]
            #print(offset)
            full.add_edge(
                "X",
                node_symbol(atoms[next_index].symbol, next_index, offset),
                bond="X:{}".format(atoms[next_index].symbol),
                ads=0
            )

        # TODO: ads also in site_graph?
        site_graph = nx.ego_graph(full, "X", radius=(radius*2)+1, distance="dist")
        site_graph = nx.subgraph(full, list(site_graph.nodes()))
        site_graph = nx.Graph(site_graph)
        full.remove_node("X")

        return site_graph

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
    ...