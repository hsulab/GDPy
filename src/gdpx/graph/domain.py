import numbers
from typing import Callable, NamedTuple

import numpy as np
from ase import Atom


class NodeID(NamedTuple):
    sym: str
    idx: int
    shift: tuple[int, int, int]


class DomainGraphFunctions(NamedTuple):
    node_id_func: Callable
    add_edge_func: Callable


def canonicalise_shift(shift: np.ndarray | tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Convert any integer-like iterable (possibly np.int32 or np.int64) into a tuple of Python ints.
    This ensures consistent hashing and equality in NetworkX.
    """
    arr = np.asarray(shift, dtype=np.int32)
    assert arr.shape == (3,)

    return int(arr[0]), int(arr[1]), int(arr[2])


def node_id_func(atom: Atom, **kwargs) -> NodeID:
    """The nodes include both local (shift=(0,0,0)) and ghost atoms."""
    index = kwargs.get("index")
    assert isinstance(index, numbers.Integral)
    shift = kwargs.get("shift")
    assert isinstance(shift, (tuple, np.ndarray))

    return NodeID(sym=atom.symbol, idx=int(index), shift=canonicalise_shift(shift))


def add_edge_func(a_i: Atom, a_j: Atom, **kwargs) -> tuple[NodeID, NodeID, dict]:
    """The ghost atoms are implicitly added when adding edges."""
    kw_i = kwargs.get("kw_i", {})
    kw_j = kwargs.get("kw_j", {})
    u = node_id_func(a_i, **kw_i)
    v = node_id_func(a_j, **kw_j)

    s_i, s_j = a_i.symbol, a_j.symbol
    bond = "{}-{}".format(*sorted([s_i, s_j]))
    edge_attrs = {"bond": bond}

    return u, v, edge_attrs


domain_graph_functions = DomainGraphFunctions(
    node_id_func=node_id_func,
    add_edge_func=add_edge_func,
)
