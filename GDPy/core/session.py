#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
from typing import NoReturn, Union, List
import pathlib

import numpy as np

from ase import Atoms

from GDPy.core.placeholder import Placeholder
from GDPy.core.variable import Variable
from GDPy.core.operation import Operation


def traverse_postorder(operation):

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)

    return nodes_postorder

class Session:

    def __init__(self, directory="./") -> NoReturn:
        """"""
        self.directory = pathlib.Path(directory)

        return

    def run(self, operation, feed_dict={}):
        """"""
        # - find forward order
        nodes_postorder = traverse_postorder(operation)
        print(nodes_postorder)

        # - set wdirs
        #for node in nodes_postorder:
        #    if hasattr(node, "directory"):
        #        node.directory = self.directory
        #        #print(node, node.directory)

        # - run nodes
        for node in nodes_postorder:
            print(f"----- {node} -----")

            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else: # Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]

                # TODO: if output exists, not forward?
                node.output = node.forward(*node.inputs)

            # - correct output type
            #if type(node.output) == list:
            #    node.output = np.array(node.output)

        return operation.output

def create_node(node_name, node_params_: dict):
    """"""
    node_params = copy.deepcopy(node_params_)
    node_type = node_params.pop("type", None)
    assert node_type is not None, f"{node_name} has no type."
    node_template = node_params.pop("template", None)
    # -- special keywords
    random_seed = node_params.pop("random_seed", None)
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        node_params.update(rng=rng)

    # -- 
    node = None
    if node_template is not None:
        node_params.update(**node_template)
    if node_type == "builder":
        from GDPy.builder.direct import FileBuilder
        node = FileBuilder(**node_params)
    elif node_type == "worker":
        from GDPy.potential.register import create_potter
        node = create_potter(**node_params)
    elif node_type == "selector":
        from GDPy.selector.interface import create_selector
        node = create_selector(node_params["selection"])
    else:
        raise RuntimeError(f"Unknown node type {node_type}.")
    # -- 
    #if not isinstance(node, Variable): # TODO
    #    node = Variable(node)

    return node

def create_operation(op_name, op_params_: dict):
    """"""
    op_params = copy.deepcopy(op_params_)
    op_type = op_params.pop("type", None)
    assert op_type is not None, f"{op_name} has no type."
    op_template = op_params.pop("template", None)
    # -- special keywords
    random_seed = op_params.pop("random_seed", None)
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        op_params.update(rng=rng)
    # --
    if op_type == "perturb":
        from GDPy.builder.perturb import perturb as op_func
    elif op_type == "drive":
        from GDPy.computation.worker.interface import drive as op_func
    elif op_type == "select":
        from GDPy.selector.interface import select as op_func
    else:
        raise RuntimeError(f"Unknown operation type {op_type}.")
    #op_params.update(func=op_func)
    #ops_dict[op_name] = op_params

    return op_func, op_params

def create_session(session_params, nodes_params, ops_params, temp_nodes, directory="./"):
    """"""
    directory = pathlib.Path(directory)

    out = None
    #session_workflow = session_config.get("session", None)
    for i, (name, params_) in enumerate(session_params.items()):
        params = copy.deepcopy(params_)
        op_type = params["type"]
        op_func, op_params = create_operation(op_type, ops_params[op_type])
        input_nodes = []
        for x in params.get("inputs", []): # TODO: step_inps maybe null
            if x in nodes_params: # variables
                inp_node = create_node(x, nodes_params[x])
            elif x in temp_nodes:
                inp_node = temp_nodes[x]
            else:
                raise RuntimeError(f"Can't find node {x}.")
            input_nodes.append(inp_node)
        out = op_func(*input_nodes, **op_params)
        if hasattr(out, "directory"):
            out.directory = directory / f"{i}_{name}"
            #print(out, out.directory)
            #if hasattr(out, "worker"):
            #    print(id(out.worker))
        (directory / f"{i}_{name}").mkdir(parents=True, exist_ok=True)
        temp_nodes[name] = out

    session = Session(directory=directory)

    return session, out

def run_session(config_filepath, directory="./"):
    """"""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    from GDPy.utils.command import parse_input_file
    session_config = parse_input_file(config_filepath)

    # - parse nodes
    nodes_params = session_config.get("nodes", None)
    ops_params = session_config.get("operations", None)
    sessions_params = session_config.get("sessions", None)

    # - create sessions
    temp_nodes = {} # intermediate nodes

    sessions = []
    for name, cur_params in sessions_params.items():
        session, end_node = create_session(
            cur_params, nodes_params, ops_params, temp_nodes,
            directory=directory/name
        )
        sessions.append([session,end_node])
    
    # - run session
    for session, end_node in sessions:
        _ = session.run(end_node, feed_dict={})

    return


if __name__ == "__main__":
    ...
