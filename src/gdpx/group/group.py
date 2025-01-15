#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import ast
import functools
import re

from ase import Atoms

from gdpx.core.register import registers
from gdpx.utils.strconv import str2list_int


def get_indices_by_index(atoms: Atoms, inp: str) -> set[int]:
    """"""
    group_indices = [int(i) for i in inp.strip().split()]

    num_atoms = len(atoms)
    for i in range(num_atoms):
        if i < 0:
            raise Exception(f"Invalid atomic index: {i+1} by {inp}.")
        if i >= num_atoms:
            raise Exception(f"Invalid atomic index: {i+1} by {inp}.")

    return set(group_indices)


def get_indices_by_id(atoms: Atoms, inp: str) -> set[int]:
    """""" ""
    group_indices = str2list_int(inp)

    num_atoms = len(atoms)
    for i in range(num_atoms):
        if i < 0:
            raise Exception(f"Invalid atomic index: {i+1} by {inp}.")
        if i >= num_atoms:
            raise Exception(f"Invalid atomic index: {i+1} by {inp}.")

    return set(group_indices)


def get_indices_by_symbol(atoms: Atoms, inp: str) -> set[int]:
    """"""
    symbols = inp.strip().split()

    group_indices = []
    for i, a in enumerate(atoms):
        if a.symbol in symbols:  # type: ignore
            group_indices.append(i)

    return set(group_indices)


def get_indices_by_tag(atoms: Atoms, inp: str) -> set[int]:
    """"""
    tag_indices = str2list_int(inp, convention="lmp", out_convention="lmp")

    tags = atoms.get_tags()
    group_indices = [i for (i, t) in enumerate(tags) if t in tag_indices]

    return set(group_indices)


def get_indices_by_region(atoms: Atoms, inp: str) -> set[int]:
    """"""
    name, *args = inp.strip().split()

    region_cls = registers.get("region", name, convert_name=True)
    region = region_cls.from_str(" ".join(args))
    group_indices = region.get_contained_indices(atoms)

    return set(group_indices)


SUPPORTED_GROUP_FUNCTIONS = dict(
    index=get_indices_by_index,
    id=get_indices_by_id,
    tag=get_indices_by_tag,
    symbol=get_indices_by_symbol,
    region=get_indices_by_region,
)


def preprocess_group_expression(grp_expr: str):
    """"""
    pattern = re.compile(r"`.*?`")

    new_grp_str, grp_funcs = grp_expr, {}
    for match in pattern.finditer(grp_expr):
        data = match.group().strip("`").split()  # remove delimiters ``
        k, v = data[0], " ".join(data[1:])
        new_grp_str = new_grp_str.replace(match.group(), k)
        if k not in SUPPORTED_GROUP_FUNCTIONS:
            raise Exception(f"Unsupported group function: {k}")
        grp_funcs[k] = functools.partial(SUPPORTED_GROUP_FUNCTIONS[k], inp=v)

    return new_grp_str, grp_funcs


def evaluate_group_expression(atoms: Atoms, grp_expr: str) -> list[int]:
    """
    Evaluate a logical expression string with and, or, not, and parentheses.
    """

    def _eval(node):
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = _eval(node.values[0])
                for value in node.values[1:]:
                    result = result.intersection(_eval(value))
                return result
            elif isinstance(node.op, ast.Or):
                result = _eval(node.values[0])
                for value in node.values[1:]:
                    result = result.union(_eval(value))
                return result
        elif isinstance(node, ast.Name):
            return grp_funcs[node.id](atoms)
        elif isinstance(node, ast.Expr):
            return _eval(node.value)
        else:
            raise TypeError(f"Unsupported type: `{type(node)}`")

    # Parse the expression into an AST
    grp_expr, grp_funcs = preprocess_group_expression(grp_expr)
    tree = ast.parse(grp_expr, mode="eval")

    group_indices = _eval(tree)

    return list(group_indices)  # type: ignore


if __name__ == "__main__":
    ...
