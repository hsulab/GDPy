#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
plt.style.use("presentation")


# Handles isomorphism for bonds
bond_match = nx.algorithms.isomorphism.categorical_edge_match("bond", "")

# Handles isomorphism for atoms with regards to perodic boundary conditions
ads_match = nx.algorithms.isomorphism.categorical_node_match(["index", "ads"], [-1, False]) 

def node_symbol(atom, offset):
    return "{}:{}:[{},{},{}]".format(atom.symbol, atom.index, offset[0], offset[1], offset[2])

def unpack_node_name(node_symbol):
    #print("xxx: ", node_symbol)
    chem_sym, idx, offset = node_symbol.split(":")
    idx = int(idx)
    offset = eval(offset)

    return chem_sym, idx, offset

def bond_symbol(atoms, a1, a2):
    return "{}{}".format(*sorted((atoms[a1].symbol, atoms[a2].symbol)))

def show_edges(graph):
    print("----- See Edges -----")
    for (u, v, d) in graph.edges.data():
        print(u, v, d)
    
    return
        
def show_nodes(graph):
    print("----- See Nodes -----")
    for (u, d) in graph.nodes.data():
        print(u, d)
    
    return

def plot_graph(graph, fig_name="graph.png"):
    # plot graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.set_title("Graph")

    nx.draw(graph, with_labels=True)

    plt.savefig(fig_name)

    return


if __name__ == "__main__":
    pass