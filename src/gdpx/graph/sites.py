#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

from typing import List

from itertools import combinations

import numpy as np
import scipy as sp
import networkx as nx

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from ..builder.group import create_an_intersect_group
from .creator import StruGraphCreator
from .comparison import bond_match, get_unique_environments_based_on_bonds
from .utils import node_symbol, unpack_node_name, show_nodes, plot_graph
from .surface import generate_normals


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

    #: Minimum inter-adsorbate distance.
    MIN_INTERADSORBATE_DISTANCE: float = 1.5

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

        # print(offsets)
        # assert len(cycle) == len(offsets), "site atomic number is inconsistent..."
        self.site_indices = site_indices
        # self.offsets = offsets
        # self.known = known

        order = len(self.site_indices)
        self.order = order

        self.position, self.normal, self.tangent = self._compute_position(
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
        # print("xxx")
        # print(names)
        # calc centre
        site_positions = np.zeros((order,3))
        for ci, n in enumerate(node_names):
            chem_sym, a_idx, offset = unpack_node_name(n)
            site_positions[ci] = atoms.positions[a_idx] + np.dot(offset, atoms.get_cell())
            # print(cycle[ci], site_positions[ci], offset)
        average = np.average(site_positions, axis=0)
        # print("site pos: ", site_positions)

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
        # if coordination == 2:
        #    average[2] = average[2] + self.shift2
        # if coordination == 3:
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
                # print(cycle)
                normal = compute_site_normal(atoms, nl, site_indices, surf_indices)
                # print(cycle,normal)
            else:
                # NOTE: order < 1 is not allowed
                ...

        for index in site_indices:
            neighbors = len(nl.get_neighbors(index)[0])
            normal += normals[index] * (1/neighbors)
        normal = normalize(normal)

        # - generate site tangent
        tangent = np.zeros(3)
        if order == 1:
            ...
        elif order == 2:
            tangent = site_positions[1] - site_positions[0]
        elif order == 3:
            tangent = site_positions[2] - (site_positions[0]+site_positions[1])/2.
        else:
            ...

        return average, normal, tangent

    def is_occupied(self, ads_elemnts):
        """"""
        ads_indices = [a.index for a in self.atoms if a.symbol in ads_elemnts]
        ads_nodes = None
        ads_nodes = [node_symbol(self.atoms[i], (0, 0, 0)) for i in ads_indices]

        ads_graphs = nx.subgraph(self.graph, ads_nodes)
        ads_graphs = [ads_graphs.subgraph(c) for c in nx.connected_components(ads_graphs)]
        print("number of adsorbate graphs: ", len(ads_graphs))
        # print(self.offsets)

        for idx, ads in enumerate(ads_graphs):
            print(f"----- adsorbate {idx} -----")
            print("ads nodes: ", ads.nodes())
            initial = list(ads.nodes())[0] # the first node in the adsorbate
            full_ads = nx.ego_graph(self.graph, initial, radius=0, distance="ads_only") # all nodes in this adsorbate, equal ads?
            # print("full ads: ", full_ads.nodes())

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
        adsorbate_, # single atom or a molecule
        other_ads_indices, # indices of other adsorbate
        ads_params: List[dict],
        check_H_bond=False
    ) -> List[Atoms]:
        """Adsorb atoms/molecule on the substrate.

        Several adsorption modes are supported.

        Args:
            ads_params: Adsorption-related parameters.

        """
        # - find indices of adsorbates
        #    NOTE: not check distance to hydrogens since
        #          since the distances are usually very small
        natoms, natoms_ads = len(self.atoms), len(adsorbate_)
        inserted_ads_indices = [i+natoms for i in range(natoms_ads) if adsorbate_[i].symbol != "H"]

        # adsorbates that are already in the substrate
        local_ads_indices = [i for i in other_ads_indices if self.atoms[i].symbol != "H"]

        # TODO: assert adsorbate to insert is a single atom or a molecule
        # -- for planar molecules such as CO2
        # adsorbate.rotate([0, 1, 0], self.normal, center=[0,0,0])
        # adsorbate.translate(self.position + (self.normal*distance_to_site))

        # -- for complex molecules

        # - start to adsorb
        ads_frames = []
        for cur_params in ads_params:
            # -- params
            mode = cur_params.get("mode", "atop")
            distance_to_site = cur_params.get("distance", 1.5)
            # -- prepare and add adsorbate
            substrate = self.atoms.copy()
            adsorbate = adsorbate_.copy()
            com = adsorbate.get_center_of_mass()
            adsorbate.translate(np.zeros(3)-com) # first translate to origin
            # --
            if mode == "atop":
                # -- for single atom or linear molecule
                adsorbate.rotate([0, 0, 1], self.normal, center=[0,0,0])
                adsorbate.translate(self.position + (self.normal*distance_to_site))
                substrate.extend(adsorbate)
            # -- more adsoprtion modes
            elif mode == "tbt": # parallel to the site
                assert self.order == 2, "Only the bridge site has the tbt adsorption."
                # NOTE: only for bridge and hollow sites
                #       default atom/molecule is along z-axis
                adsorbate.rotate([0, 0, 1], self.tangent, center=[0,0,0])
                adsorbate.translate(self.position + (self.normal*distance_to_site))
                substrate.extend(adsorbate)
            elif mode == "thb":
                assert self.order == 3, "Only the hollow site has the thb adsorption."
                adsorbate.rotate([0, 0, 1], self.tangent, center=[0,0,0])
                adsorbate.translate(self.position + (self.normal*distance_to_site))
                substrate.extend(adsorbate)
            else:
                raise NotImplementedError(f"Adsorption mode {mode} is not supported.")

            # - check distances between adsorbates
            dist = float("inf")
            if len(local_ads_indices) != 0:
                for i in inserted_ads_indices:
                    dists = substrate.get_distances(i, local_ads_indices, mic=True)
                    dist = min(dist, dists.min())
                # TODO: check is_occupied
                if dist < self.MIN_INTERADSORBATE_DISTANCE:
                    substrate = None
            if substrate is not None: 
                ads_frames.append(substrate)

        return ads_frames

    def _iadsorb(self):
        """"""

        return


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

    """Find adsorption sites on surface.

    The procedure includes several steps: 1. Detect surface atoms. 2. Find site 
    with different orders (coordination number). This supports both mono- and 
    multi-dentate sites.

    """

    #: The minimum norm to be considered as surface.
    surface_normal: float = 0.65

    _site_radius = 2

    # bind an atoms object
    atoms: Atoms = None

    nl = None
    graph: nx.Graph = None

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
    
    def find(self, atoms_: Atoms, site_params: list, surface_mask: List[int]=None):
        """Find all sites based on graph.

        The site finding includes several steps. 1. Create a graph representation of 
        the structure. 2. Determine the surface normal. 3. Determine valid sites.

        Args:
            site_params: Parameters that define the site to find.

        """
        # - input atoms
        atoms = atoms_

        # - find adsorbate atoms' indices
        ads_indices = [a.index for a in atoms if a.symbol in self.adsorbate_elements]

        # - create graph
        graph = self.generate_graph(atoms, ads_indices) # create self.graph

        # NOTE: update nl since we want use nl with bothways instead of false
        nl = self.neigh_creator.build_neighlist(atoms, bothways=True)
        nl.update(atoms)

        # TODO: surface region group command?
        # - find atoms that are on surface and possibly form adsorption sites...
        # NOTE: make sure to manually set the normals for 2-D materials, 
        #       all atoms should have a normal pointing up, as all atoms are surface atoms
        system = self.check_system(atoms)

        self._print(f"{atoms.pbc = }")
        self._print(f"{system = }")

        normals, surface_mask = generate_normals(
            atoms, nl, surface_mask, 
            ads_indices, mask_elements=[],
            surf_norm_min=self.surface_normal, normalised=True,
            system=system
        )
        self._print(f"number of surface atoms: {len(surface_mask)}")
        self._debug(f"{surface_mask = }")
        self._debug(f"surface mask ads: {ads_indices}")

        atoms.arrays["surface_direction"] = normals

        # - find all valid adsorption sites
        site_groups = []
        for i, curr_site_params in enumerate(site_params):
            self._print(f"site type {i}: {curr_site_params}")
            curr_params_list = self._broadcast_site_params(curr_site_params)
            #print(cur_params_list)

            nanchors = len(curr_params_list)

            if nanchors == 1:
                self._print("Monodentate sites...")
                curr_params = curr_params_list[0]
                sites = self._generate_single_site(
                    curr_params, atoms, graph, nl, surface_mask, normals
                )
                ...
            else:
                self._print("Multidentate sites...")
                #for curr_params in curr_params_list:
                #    ...
                raise NotImplementedError()
            
            site_groups.append(sites)

        return site_groups
    
    def _generate_single_site(
        self, params: dict, atoms: Atoms, graph, nl, surface_mask, normals
    ):
        """Generate single-multidentate site."""
        # - some basic params
        cn = params.get("cn") # coordination number
        cur_site_radius = params.get("radius", self._site_radius)
        self._print(f"coordination number: {cn}")

        # - find possible atoms to form the site
        self._print(f"graph: {graph}")
        #cur_species = params.get("species", None)
        #site_indices = params.get("site_indices", None)
        #region = params.get("region", None)
        #cur_surf_mask = find_valid_indices(
        #    atoms, cur_species, spec_indices=site_indices, region=region
        #)
        natoms = len(atoms)
        group_commands = params.get(
            "group", 
            # default: return indices of all atoms
            ["id {}".format(" ".join([str(i) for i in range(1,natoms+1)]))] # start from 1
        )
        valid_indices = create_an_intersect_group(atoms, group_commands)

        # -- Site indices must be in the surface indices
        valid_indices = [i for i in valid_indices if i in surface_mask]

        self._print(f"number of site atoms: {len(valid_indices)}")
        self._debug(f"atomic indices for site: {valid_indices}")

        # -- create sites
        found_sites = self._generate_site(
            atoms, graph, nl,
            valid_indices, surface_mask, normals, coordination=cn
        )

        # - generate site graph TODO: move this to site object?
        for s in found_sites:
            s.graph = self.process_site(
                atoms, graph, nl, s.site_indices, cur_site_radius
            )
        #print(found_sites)
        
        if found_sites:
            # -- get sites with unique environments
            #print("\n\nfound sites: ", len(found_sites))
            #for s in found_sites:
            #    print(s, s.graph)
            site_graphs = [s.graph for s in found_sites]
            unique_indices = get_unique_environments_based_on_bonds(site_graphs)
            #print(unique_indices)
            #print(unique_indices)
            unique_sites = [found_sites[i] for i in unique_indices]
        
            # TODO: sort sites by positions?
            self._print(f"unique sites: {len(found_sites)} -> {len(unique_sites)}")
        else:
            unique_sites = []
            self._print("No sites are found.")

        return unique_sites
    
    def _broadcast_site_params(self, params: dict) -> List[dict]:
        """"""
        # params need to broadcast
        keys = ["cn", "radius", "group", "distance"]
        nums = []
        for key in keys:
            cur_num = 1
            if key == "cn": # int
                # coordination number of the site
                value = params.get(key, 1)
                if isinstance(value, list):
                    cur_num = len(value)
            elif key == "radius": # int
                value = params.get(key, 3)
                if isinstance(value, list):
                    cur_num = len(value)
                    num = cur_num
            elif key == "group": # List[str]
                # NOTE: this param is not broadcastable
                #value = params.get(key, None)
                #if isinstance(value, list): # all atoms
                #    cur_num = len(value)
                #    num = cur_num
                ...
            elif key == "distance": # float
                value = params.get(key, 1.5)
                if isinstance(value, list):
                    cur_num = len(value)
                    num = cur_num
            else:
                ...
            nums.append(cur_num)

        # number of sites for a single adsorption
        num = max(nums) # consider a single-dentate site as default if num == 1 
        for key, cur_num in zip(keys, nums):
            assert (cur_num == 1 or cur_num == num), f"Parameter {key} size is inconsistent."
        
        # - create new parameters
        params_list = []
        for i in range(num):
            cur_params = dict()
            for key in keys:
                if key == "cn":
                    # coordination number of the site
                    value = params.get(key, 1)
                    if isinstance(value, list):
                        cur_params[key] = value[i]
                    else:
                        cur_params[key] = value
                elif key == "radius":
                    value = params.get(key, 3)
                    if isinstance(value, list):
                        cur_params[key] = value[i]
                    else:
                        cur_params[key] = value
                elif key == "group":
                    # NOTE: this param is not broadcastable
                    value = params.get(key, None)
                    #if isinstance(value, list): # all atoms
                    #    cur_num = len(value)
                    #    num = cur_num
                    cur_params[key] = value
                elif key == "distance":
                    value = params.get(key, 1.5)
                    if isinstance(value, list):
                        cur_params[key] = value[i]
                    else:
                        cur_params[key] = value
                else:
                    ...
            params_list.append(cur_params)

        return params_list
 
    def _generate_site(
        self, 
        atoms: Atoms,
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
        self._debug(f"DEBUG: {valid}")

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
                radius=self.DIS_SURF2SURF, center=True, # 2 is surf_surf_dis
                distance="dist"
            ) # go over framework
            #plot_graph(local_graph, f"local-{coordination}.png")

            site_node_names = [] # List[List[str]]
            if coordination == 3:
                #print("node names: ", local_graph.nodes())
                all_cliques = nx.enumerate_all_cliques(local_graph)
                site_node_names = [x for x in all_cliques if len(x)==3]
            # TODO: add CN4 site that is a square
            else:
                # NOTE: for very small unit cell, 
                #       nodes maybe connected by their neighbour images
                #       e.g. bridge sites on p(2x2)-Pt(111)
                nodes_comb = combinations(local_graph.nodes, coordination)
                new_site_graphs = []
                for node_names in nodes_comb:
                    if coordination == 2:
                        # check they are connected
                        if graph.has_edge(node_names[0],node_names[1]):
                            new_site_graphs.append(nx.subgraph(graph, node_names))
                    else:
                        # top site
                        new_site_graphs.append(nx.subgraph(graph, node_names))

                # *** check if node number is valid
                for g in new_site_graphs:
                    node_names = list(g.nodes())
                    assert len(node_names) == coordination, "The site order is incorrect."
                    site_node_names.append(node_names)

            # --- create site object ---
            #self.pfunc(site_node_names)
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


if __name__ == "__main__":
    ...
