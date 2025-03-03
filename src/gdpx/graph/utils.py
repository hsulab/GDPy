#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union

import matplotlib.pyplot as plt
import networkx as nx


def node_symbol(symbol, idx, offset):
    """Create a node name from its components."""
    return "{}:{}:[{},{},{}]".format(symbol, idx, offset[0], offset[1], offset[2])


def unpack_node_name(node_symbol):
    """Unpack the node name into its components."""
    chem_sym, idx, offset = node_symbol.split(":")
    idx = int(idx)
    offset = eval(offset)

    return chem_sym, idx, offset


def bond_symbol(sym1, sym2, a1, a2):
    """Create a bond name from its components."""
    return "{}{}".format(*sorted((sym1, sym2)))


def grid_iterator(grid: Union[int, tuple[int, int, int]]):
    """Yield all of the coordinates in a 3D grid as tuples.

    Args:
        grid (tuple[int] or int): The grid dimension(s) to
                                  iterate over (x or (x, y, z))

    Yields:
        tuple: (x, y, z) coordinates
    """
    # Expand to 3D grid
    if isinstance(grid, int):
        grid = (grid, grid, grid)

    for x in range(-grid[0], grid[0] + 1):
        for y in range(-grid[1], grid[1] + 1):
            for z in range(-grid[2], grid[2] + 1):
                yield (x, y, z)


def show_edges(graph):
    print("----- See Edges -----")
    for u, v, d in graph.edges.data():
        print(u, v, d)

    return


def show_nodes(graph):
    print("----- See Nodes -----")
    for u, d in graph.nodes.data():
        print(u, d)

    return


def show_components(graph):
    """Show the connected components of the graph."""
    print("----- connected components -----")
    for c in nx.connected_components(graph):
        print(c)

    return


def plot_graph(graph, fig_name="graph.png"):
    """Plot the graph."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    ax.set_title("Graph")  # type: ignore

    nx.draw(graph, with_labels=True)

    fig.savefig(fig_name, bbox_inches="tight")

    return


if __name__ == "__main__":
    ...

