#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import ast
import functools
import re
from typing import Literal, Optional

import numpy as np
from ase import Atoms

from gdpx.utils.strconv import string_to_integers
from gdpx.structures.geometry.wrapping import validate_wrap_interval, wrap_fractional


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
    group_indices = string_to_integers(inp, convention="lmp")

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
    tag_indices = string_to_integers(inp, convention="lmp", out_convention="lmp")

    tags = atoms.get_tags()
    group_indices = [i for (i, t) in enumerate(tags) if t in tag_indices]

    return set(group_indices)


def get_indices_by_pos(atoms: Atoms, inp: str, component: Literal["x", "y", "z"], reverse: bool) -> set[int]:
    """Get indices by fractional coordinates along the z-axis.

    Note:
        The fractional coordinates are wrapped to [-0.1, 0.9) by default.

    Args:
        atoms: The ASE atoms object.
        inp: The group expression.
        component: The component to use, must be one of "x", "y", or "z".
        reverse: If True, the top atoms are selected, otherwise the bottom atoms.

    Examples:
        >>> get_indices_by_pos(atoms, "16")  # The bottom 16 atoms.
        >>> get_indices_by_pos(atoms, "16 0.0 1.0")  # The frac is wrapped to [0.0, 1.0).

    Returns:
        The integer indices.

    """
    fields = inp.strip().split()
    if len(fields) not in (1, 3):
        raise ValueError("Position group syntax: count [wrap_lower wrap_upper].")
    number = int(fields[0])
    if number < 0:
        raise ValueError("Position group count must be non-negative.")
    bounds = (-0.1, 0.9) if len(fields) == 1 else tuple(float(x) for x in fields[1:])
    bounds = validate_wrap_interval(bounds)

    axis = {"x": 0, "y": 1, "z": 2}[component]
    atomic_indices = list(range(len(atoms)))
    if atoms.pbc[axis]:
        fractional = atoms.get_scaled_positions(wrap=False)[:, axis]
        wrapped_coordinates = wrap_fractional(fractional, bounds)
    else:
        wrapped_coordinates = atoms.positions[:, axis]

    group_indices = sorted(atomic_indices, key=lambda x: wrapped_coordinates[x], reverse=reverse)[:number]

    return set(group_indices)


def get_indices_by_region(atoms: Atoms, inp: str) -> set[int]:
    """"""
    name, *args = inp.strip().split()

    from gdpx.structures.regions import REGISTER as region_registry

    class_name = "".join(part.capitalize() for part in name.split("_")) + "Region"
    region_cls = region_registry[class_name]
    region = region_cls.from_str(inp)  # The string should be "{name} {*args}"
    group_indices = region.get_contained_indices(atoms)

    return set(group_indices)


SUPPORTED_GROUP_FUNCTIONS = dict(
    index=get_indices_by_index,
    id=get_indices_by_id,
    tag=get_indices_by_tag,
    symbol=get_indices_by_symbol,
    zbot=functools.partial(get_indices_by_pos, component="z", reverse=False),
    ztop=functools.partial(get_indices_by_pos, component="z", reverse=True),
    region=get_indices_by_region,
)


def preprocess_group_expression(grp_expr: str) -> tuple[str, list[functools.partial]]:
    """Tokenize selectors and compile only boolean structure to a Python AST.

    Selector arguments end at a boolean operator or parenthesis. They are passed
    to the existing selector parsers, never interpreted as Python expressions.
    """
    if not isinstance(grp_expr, str) or not grp_expr.strip():
        raise ValueError("A group expression must be a nonempty string.")
    if "`" in grp_expr:
        raise ValueError("Remove backticks from group expressions; use e.g. not tag 0.")
    if re.search(r"[=!<>]", grp_expr):
        raise ValueError("Literal comparisons are unsupported; use selectors such as not tag 0.")
    tokens = re.findall(r"\(|\)|[^\s()]+", grp_expr)
    compiled, functions = [], []
    position = 0
    boundaries = {"and", "or", "not", "(", ")"}
    while position < len(tokens):
        token = tokens[position]
        position += 1
        if token in boundaries or token in {"active", "buffer", "frozen"}:
            compiled.append(token)
            continue
        if token not in SUPPORTED_GROUP_FUNCTIONS:
            raise ValueError(f"Unknown group selector: {token}")
        arguments = []
        while position < len(tokens) and tokens[position] not in boundaries:
            arguments.append(tokens[position])
            position += 1
        if not arguments:
            raise ValueError(f"Group selector {token} requires arguments.")
        compiled.append(f"group_{len(functions)}")
        functions.append(functools.partial(SUPPORTED_GROUP_FUNCTIONS[token], inp=" ".join(arguments)))
    return " ".join(compiled), functions


def evaluate_group_expression(atoms: Atoms, grp_expr: Optional[str] = None) -> list[int]:
    """
    Evaluate a logical expression string with and, or, not, and parentheses.
    """

    def _eval(node):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return set(range(len(atoms))) - _eval(node.operand)
        elif isinstance(node, ast.BoolOp):
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
            if node.id in {"active", "buffer", "frozen"}:
                array = atoms.arrays.get("gdpx_" + node.id)
                if array is None:
                    array = atoms.info.get("data", {}).get("search_space", {}).get(node.id)
                if array is None or len(array) != len(atoms):
                    raise ValueError(f"Named group {node.id} requires resolved system.search_space membership.")
                return set(np.flatnonzero(array).tolist())
            match = re.fullmatch(r"group_(\d+)", node.id)
            if match is None or int(match[1]) >= len(grp_funcs):
                raise ValueError(f"Unknown group name: {node.id}")
            return grp_funcs[int(match[1])](atoms)
        elif isinstance(node, ast.Expression):
            return _eval(node.body)
        else:
            raise TypeError(f"Unsupported type: {node} of {type(node)}")

    if grp_expr is not None:
        # Parse the expression into an AST
        grp_expr, grp_funcs = preprocess_group_expression(grp_expr)
        try:
            tree = ast.parse(grp_expr, mode="eval")
        except SyntaxError as error:
            raise ValueError("Invalid group expression; combine selectors with and, or, not, and parentheses.") from error
        group_indices = _eval(tree)
    else:
        # Fall back to all atoms
        num_atoms = len(atoms)
        group_indices = range(num_atoms)

    return list(group_indices)  # type: ignore


if __name__ == "__main__":
    ...
