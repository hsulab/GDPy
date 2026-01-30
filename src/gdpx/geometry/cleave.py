from ase import Atoms

from gdpx.group import evaluate_group_expression


def cleave_structures_by_group(structures: list[Atoms], grp_expr: str) -> tuple[list[Atoms], list[int]]:
    """"""
    new_structures, mapping_indices = [], []
    for i, atoms in enumerate(structures):
        group_indices = evaluate_group_expression(atoms, grp_expr)
        if group_indices:
            cleaved = atoms[group_indices]
            new_structures.append(cleaved)
            mapping_indices.append(i)
        else:
            ...

    return new_structures, mapping_indices
