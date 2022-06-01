#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx

from GDPy.graph.utils import node_symbol, bond_symbol, bond_match, ads_match
from GDPy.graph.utils import show_nodes, show_edges

class AdsSite(object):

    MIN_INTERADSORBATE_DISTANCE = 1.5

    def __init__(
        self, atoms, 
        cycle, # indices of site atoms
        #offsets,
        #known,
        position, 
        normal, 
        graph=None,
        **kwargs
    ):
        """"""
        self.atoms = atoms

        #print(offsets)
        #assert len(cycle) == len(offsets), "site atomic number is inconsistent..."
        self.cycle = cycle
        #self.offsets = offsets
        #self.known = known
        
        self.position = position

        self.normal = normal
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
        ).format(self.cycle, *self.position, *self.normal, bool(self.graph))
        return content
    
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