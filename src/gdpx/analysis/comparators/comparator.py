import numpy as np
from ase import Atoms

from gdpx.core.component import BaseComponent
from gdpx.structures.groups import evaluate_group_expression


class BaseComparator(BaseComponent):
    def _prepare_groups(self, structures):
        """Copy and canonicalize selected groups, checking their compositions."""
        groups = []
        for atoms in structures:
            indices = sorted(evaluate_group_expression(atoms, self.group_expression))
            if not indices:
                raise ValueError("group_expression selected an empty comparison group.")
            group = atoms[indices]
            group = group[np.argsort(group.numbers, kind="stable")]
            if not np.all(np.isfinite(group.positions)):
                raise ValueError("Comparison positions must be finite.")
            if groups and not np.array_equal(groups[0].numbers, group.numbers):
                raise ValueError("Comparison groups must have identical compositions.")
            groups.append(group)
        return groups

    def compare_composition(self, a1: Atoms, a2: Atoms) -> bool:
        """Compare two atoms based on number of atoms, chell, and chemical formula sequentially.

        Note:
            The periodic boundary conditions (PBC) are not considered.

        """
        is_similar = False
        na1, na2 = len(a1), len(a2)
        if na1 == na2:
            c1, c2 = a1.get_cell(complete=True), a2.get_cell(complete=True)
            if np.allclose(c1, c2):
                s1, s2 = a1.get_chemical_formula(), a2.get_chemical_formula()
                if s1 == s2:
                    is_similar = True
            else:
                ...
        else:
            ...

        return is_similar

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """"""
        is_similar = self.compare_composition(a1, a2)

        return is_similar

    def __call__(self, a1, a2) -> bool:
        """"""

        return self.looks_like(a1, a2)
