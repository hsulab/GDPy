from typing import Optional

import networkx as nx
from ase import Atoms
from joblib import Parallel, delayed

from gdpx.graph.base import AtomicGraph
from gdpx.group import evaluate_group_expression
from gdpx.utils.profiler import CustomTimer

from .comparator import BaseComparator

bond_match = nx.algorithms.isomorphism.categorical_edge_match("bond", "")


class GraphComparator(BaseComparator):
    def __init__(
        self, group: Optional[str] = None, ignored_pairs: Optional[list[tuple[str, str]]] = None, *args, **kwargs
    ):
        """Initialise the comparator.

        Args:
            group: The group expression to select atoms to build graph for comparison.

        """
        super().__init__(*args, **kwargs)

        self.group = group

        ignore_pairs_ = []
        if ignored_pairs is not None:
            for s1, s2 in ignored_pairs:
                ignore_pairs_.append("-".join([s1, s2]))
                ignore_pairs_.append("-".join([s2, s1]))
        self.ignored_pairs = ignore_pairs_

        return

    @staticmethod
    def _process_single_structure(atoms: Atoms, group: Optional[str], ignored_pairs: list[str]) -> nx.Graph:
        """"""
        group_indices = evaluate_group_expression(atoms, group)

        graph_builder = AtomicGraph(atoms, graph_type="partial")
        graph_builder.build(
            indices=group_indices, ratio=1.0, skin=0.2, include_neighbors=True, ignored_bonds=ignored_pairs
        )
        graph = graph_builder.graph
        assert isinstance(graph, nx.Graph)

        return graph

    def prepare_data(self, frames: list[Atoms]):
        """"""
        with CustomTimer(name="creating graphs", func=self._print):
            graphs = Parallel(n_jobs=self.njobs)(
                delayed(self._process_single_structure)(atoms, self.group, self.ignored_pairs) for atoms in frames
            )

        return graphs

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """"""
        is_similar = super().looks_like(a1, a2)
        if is_similar:
            fingerprints = self.prepare_data([a1, a2])
            fp1, fp2 = fingerprints
            is_similar = self.compare_fingerprints(fp1, fp2)

        return is_similar

    def compare_fingerprints(self, fp1, fp2):
        """"""
        is_isomorphic = nx.algorithms.isomorphism.is_isomorphic(fp1, fp2, edge_match=bond_match)

        return is_isomorphic
