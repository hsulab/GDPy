#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Any, Optional

from gdpx import config
from gdpx.core.register import registers
from gdpx.session.operation import Operation


def traverse_postorder(operation: Operation):

    nodes_postorder = []
    identifiers = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        if id(node) not in identifiers:
            nodes_postorder.append(node)
            identifiers.append(id(node))

    recurse(operation)

    return nodes_postorder


def create_variable(node_name: Optional[str], node_params_: Any):
    """Create a variable from registers."""
    node_params = copy.deepcopy(node_params_)
    node_type = node_params.pop("type", None)
    assert node_type is not None, f"{node_name} has no type."
    node_template = node_params.pop("template", None)
    config._debug(node_name)
    config._debug(node_params)

    node = None
    if node_template is not None:
        node_params.update(**node_template)
    node_cls = registers.get("variable", node_type, convert_name=True)
    node = node_cls(**node_params)

    return node


def create_operation(op_name: Optional[str], op_params_: Any):
    """Create an operation from registers."""
    op_params = copy.deepcopy(op_params_)
    op_type = op_params.pop("type", None)
    assert op_type is not None, f"{op_name} has no type."
    _ = op_params.pop("template", None)  # Use template?
    config._debug(op_name)
    config._debug(op_params)

    op_cls = registers.get("operation", op_type, convert_name=False)
    operation = op_cls(**op_params)

    return operation


if __name__ == "__main__":
    ...
