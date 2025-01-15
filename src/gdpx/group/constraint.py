#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
from typing import Optional

from ase.constraints import FixAtoms, constrained_indices

from .group import evaluate_group_expression


def canonicalise_constraint_expression(cons_expr: str) -> str:
    """Convert a constraint expression to a canonical form for compatibility."""
    cons_expr = cons_expr.strip()
    if "`" in cons_expr:
        new_cons_expr = cons_expr
    elif "lowest" in cons_expr:
        new_cons_expr = "`" + "zbot " + cons_expr.split()[1] + "`"
    else:
        if re.match(r"^\d.*\d$", cons_expr):
            new_cons_expr = "`" + "id " + cons_expr + "`"
        else:
            raise Exception(f"Invalid constraint expression: {cons_expr}.")

    return new_cons_expr


def evaluate_constraint_expression(
    atoms, cons_expr: Optional[str], ignore_ase_constraints=True
) -> tuple[list[int], list[int]]:
    """"""
    atomic_indices = list(range(len(atoms)))
    if cons_expr is not None:
        cons_expr = canonicalise_constraint_expression(cons_expr)
        frozen_indices = evaluate_group_expression(atoms, cons_expr)
    else:
        frozen_indices = []

    mobile_indices = [i for i in atomic_indices if i not in frozen_indices]

    return mobile_indices, frozen_indices


if __name__ == "__main__":
    ...
