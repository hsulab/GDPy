#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
from typing import NoReturn, Union, List
import pathlib

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.core.placeholder import Placeholder
from GDPy.core.variable import Variable
from GDPy.core.operation import Operation
from GDPy.core.register import registers


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

    def run(self, operation, feed_dict={}) -> NoReturn:
        """"""
        # - find forward order
        nodes_postorder = traverse_postorder(operation)
        #print(nodes_postorder)

        # - run nodes
        for node in nodes_postorder:
            print(node)
            print(f"----- Node {node.__class__.__name__} @ {node.directory.name} -----")

            if isinstance(node, Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else: # Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]

                # TODO: if output exists, not forward?
                node.output = node.forward(*node.inputs)
                if node.status != "finished":
                    print(f"node is still forwarding! Try later!")
                    break

        return

def create_placeholder(node_name, node_params_: dict):
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
    node_cls = registers.get("placeholder", node_type, convert_name=True)
    node = node_cls(**node_params)

    return node

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
    random_seed = op_params.pop("random_seed", None)
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        op_params.update(rng=rng)
    # --
    op_method = op_params.pop("method", None)
    op_func = registers.get("operation", op_type, convert_name=False)

    return op_func, op_params

def create_session(session_params, phs_params, nodes_params, ops_params, temp_nodes, directory="./"):
    """"""
    directory = pathlib.Path(directory)

    out, placeholders = None, []
    #session_workflow = session_config.get("session", None)
    for i, (name, params_) in enumerate(session_params.items()):
        params = copy.deepcopy(params_)
        op_type = params["type"]
        op_func, op_params = create_operation(op_type, ops_params[op_type])
        input_nodes = []
        for x in params.get("inputs", []): # TODO: step_inps maybe null
            if x in phs_params: # placeholders
                inp_node = create_placeholder(x, phs_params[x])
                placeholders.append(inp_node)
            elif x in nodes_params: # variables
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
        #(directory / f"{i}_{name}").mkdir(parents=True, exist_ok=True)
        # -- check if has duplicated node names!!!
        if name not in temp_nodes:
            temp_nodes[name] = out
        else:
            raise RuntimeError(f"Found duplicated node {name} in session.")

    session = Session(directory=directory)

    return session, out, placeholders

def run_session(config_filepath, custom_session_names=None, entry_string: str=None, directory="./"):
    """Run a session based on user-defined input.

    Read definitions of nodes and operations from the file and placeholders from
    the command line.

    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # - parse entry data for placeholders
    entry_data = {}
    if entry_string is not None:
        data = entry_string.strip().split()
        key_starts = []
        for i, d in enumerate(data):
            if d in ["structure"]:
                key_starts.append(i)
        key_ends = key_starts[1:] + [len(data)]
        for s, e in zip(key_starts, key_ends):
            curr_name = data[s]
            if curr_name == "structure":
                curr_data = read(data[s+1:e][0], ":")
            entry_data[curr_name] = curr_data
    print("entry", entry_data)

    # - parse nodes
    from GDPy.utils.command import parse_input_file
    session_config = parse_input_file(config_filepath)

    phs_params = session_config.get("placeholders", None)
    nodes_params = session_config.get("nodes", None)
    ops_params = session_config.get("operations", None)
    sessions_params = session_config.get("sessions", None)

    # - create sessions
    temp_nodes = {} # intermediate nodes, shared among sessions

    sessions = {}
    for name, cur_params in sessions_params.items():
        sessions[name] = create_session(
            cur_params, phs_params, nodes_params, ops_params, temp_nodes,
            directory=directory/name
        )
    
    # - run session
    if custom_session_names is None:
        custom_session_names = copy.deepcopy(list(sessions.keys()))
    for name, (session, end_node, placeholders) in sessions.items():
        feed_dict = {p:entry_data[p.name] for p in placeholders}
        if name in custom_session_names:
            print(f"===== run session {name} =====")
            _ = session.run(end_node, feed_dict=feed_dict)

    return


if __name__ == "__main__":
    ...
