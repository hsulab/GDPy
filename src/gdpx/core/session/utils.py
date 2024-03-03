#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

from .. import config
from ..register import registers
from ..operation import Operation

def traverse_postorder(operation):

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

def create_variable(node_name, node_params_: dict):
    """"""
    node_params = copy.deepcopy(node_params_)
    node_type = node_params.pop("type", None)
    assert node_type is not None, f"{node_name} has no type."
    node_template = node_params.pop("template", None)
    # -- special keywords
    #random_seed = node_params.pop("random_seed", None)
    #if random_seed is not None:
    #    rng = np.random.default_rng(random_seed)
    #    node_params.update(rng=rng)
    config._debug(node_name)
    config._debug(node_params)

    # -- 
    node = None
    if node_template is not None:
        node_params.update(**node_template)
    node_cls = registers.get("variable", node_type, convert_name=True)
    node = node_cls(**node_params)

    return node

def create_operation(op_name, op_params_: dict):
    """"""
    op_params = copy.deepcopy(op_params_)
    op_type = op_params.pop("type", None)
    assert op_type is not None, f"{op_name} has no type."
    op_template = op_params.pop("template", None)
    # -- special keywords
    #random_seed = op_params.pop("random_seed", None)
    #if random_seed is not None:
    #    rng = np.random.default_rng(random_seed)
    #    op_params.update(rng=rng)
    config._debug(op_name)
    config._debug(op_params)

    # --
    op_cls = registers.get("operation", op_type, convert_name=False)
    operation = op_cls(**op_params)

    return operation


if __name__ == "__main__":
    ...
