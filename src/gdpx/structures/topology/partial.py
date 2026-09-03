import numbers
from typing import Callable, NamedTuple

from ase import Atom


class PartialGraphFunctions(NamedTuple):
    node_id_func: Callable
    add_edge_func: Callable


def node_id_func(atom: Atom, **kwargs) -> str:
    """Generate a node ID string for an atom."""
    # No information of ghost atoms is stored in the graph!!
    index = kwargs.get("index")
    assert isinstance(index, numbers.Integral)

    return f"{atom.symbol}_{int(index)}"


def add_edge_func(a_i: Atom, a_j: Atom, **kwargs) -> tuple[str, str, dict]:
    """Generate edge information between two atoms."""
    # If the box is too small, the edge between two atoms may appear multiple times with different shifts,
    # which are not considered here.
    kw_i = kwargs.get("kw_i", {})
    kw_j = kwargs.get("kw_j", {})
    u = node_id_func(a_i, **kw_i)
    v = node_id_func(a_j, **kw_j)

    s_i, s_j = a_i.symbol, a_j.symbol
    bond = "{}-{}".format(*sorted([s_i, s_j]))
    shift_j = kw_j.get("shift")
    edge_attrs = {"bond": bond, "shift": shift_j}

    return u, v, edge_attrs


partial_graph_functions = PartialGraphFunctions(
    node_id_func=node_id_func,
    add_edge_func=add_edge_func,
)
