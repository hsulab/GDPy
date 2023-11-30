#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import networkx as nx

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
#plt.style.use("presentation")

def node_symbol(symbol, idx, offset):
    """"""
    return "{}:{}:[{},{},{}]".format(symbol, idx, offset[0], offset[1], offset[2])

def unpack_node_name(node_symbol):
    #print("xxx: ", node_symbol)
    chem_sym, idx, offset = node_symbol.split(":")
    idx = int(idx)
    offset = eval(offset)

    return chem_sym, idx, offset

def bond_symbol(sym1, sym2, a1, a2):
    return "{}{}".format(*sorted((sym1, sym2)))

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

def show_components():
    #print("----- connected components -----")
    #for c in nx.connected_components(graph):
    #    print(c)
    pass

    return

def plot_graph(graph, fig_name="graph.png"):
    # plot graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.set_title("Graph")

    nx.draw(graph, with_labels=True)

    plt.savefig(fig_name)

    return

if __name__ == "__main__":
    ...