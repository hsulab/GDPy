#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import json
from typing import NoReturn, Union, List, Callable
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

    _print: Callable = print
    _debug: Callable = print

    def __init__(self, directory="./") -> NoReturn:
        """"""
        self.directory = pathlib.Path(directory)

        return

    def run(self, operation, feed_dict: dict={}) -> NoReturn:
        """"""
        # - find forward order
        nodes_postorder = traverse_postorder(operation)
        self._print(
            "[{:^24s}] NUM_NODES: {} AT MAIN: {}".format(
                "START", len(nodes_postorder), str(self.directory)
            )
        )

        # - run nodes
        for i, node in enumerate(nodes_postorder):
            # NOTE: reset directory since it maybe changed
            prev_name = node.directory.name
            if not prev_name:
                prev_name = node.__class__.__name__
            node.directory = self.directory/f"{str(i).zfill(4)}.{prev_name}"
            if node.__class__.__name__.endswith("Variable"):
                node_type = "VX"
            else:
                node_type = "OP"
            self._print(
                "[{:^24s}] NAME: {} AT {}".format(
                    node_type, node.__class__.__name__.upper(), node.directory.name
                )
            )

            if isinstance(node, Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else: # Operation
                if node.preward():
                    node.inputs = [input_node.output for input_node in node.input_nodes]
                    node.output = node.forward(*node.inputs)
                    #if node.status != "finished":
                    #    print(f"node is still forwarding! Try later!")
                    #    break
                else:
                    print("wait previous nodes to finish...")
                    continue

        return
    
class CyclicSession:

    """Create a cyclic session.

    This supports a session that contains preprocess, iteration, and postprocess.

    """

    _print: Callable = print
    _debug: Callable = print

    def __init__(self, directory="./") -> None:
        """"""
        self.directory = pathlib.Path(directory)

        return
    
    def iteration(self, operations):
        """Iterative part that converges at certain criteria.

        This noramlly includes steps: sample, select, label, and train. Some inputs of
        operations should be update during the iterations, for instance, the models 
        in the potential.

        """

        return
    
    def run(self, init_node, iter_node=None, post_node=None, repeat=1, *args, **kwargs) -> NoReturn:
        """"""
        # - init
        self._print(("="*28+"{:^24s}"+"="*28+"\n").format("INIT"))
        session = Session(self.directory/"init")
        _ = session.run(init_node, feed_dict={})
        self._print("status: ", init_node.status)

        init_state = init_node.status
        if init_state != "finished":
            self._print("wait init session to finish...")
            return

        # - iter
        curr_potter_node = init_node # a node that forwards a potter manager
        for i in range(repeat):
            self._print(("="*28+"{:^24s}"+"="*28+"\n").format(f"ITER.{str(i).zfill(4)}"))
            session = Session(self.directory/f"iter.{str(i).zfill(4)}")
            # -- update some parameters
            curr_node = copy.deepcopy(iter_node)
            nodes_postorder = traverse_postorder(curr_node)
            # --- trainer
            # --- potter
            for x in nodes_postorder:
                if x.__class__.__name__ == "drive":
                    x.input_nodes[1]._update_workers(curr_potter_node)
            # -- run
            _ = session.run(curr_node, feed_dict={})
            curr_state = curr_node.status
            if curr_state != "finished":
                self._print(f"wait iter.{str(i).zfill(4)} session to finish...")
                break
            else:
                curr_potter_node = curr_node
        else:
            self._print(f"iter finished...")

        # - post

        return

    def _irun(self, operation, feed_dict: dict={}) -> NoReturn:
        """"""
        # - find forward order
        nodes_postorder = traverse_postorder(operation)
        self._print(
            "[{:^24s}] NUM_NODES: {} AT MAIN: {}".format(
                "START", len(nodes_postorder), str(self.directory)
            )
        )

        # - run nodes
        for i, node in enumerate(nodes_postorder):
            # NOTE: reset directory since it maybe changed
            prev_name = node.directory.name
            if not prev_name:
                prev_name = node.__class__.__name__
            node.directory = self.directory/f"{str(i).zfill(4)}.{prev_name}"
            if node.__class__.__name__.endswith("Variable"):
                node_type = "VX"
            else:
                node_type = "OP"
            self._print(
                "[{:^24s}] {:<36s} AT {}".format(
                    node_type, node.__class__.__name__.upper(), node.directory.name
                )
            )

            if isinstance(node, Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else: # Operation
                if node.preward():
                    node.inputs = [input_node.output for input_node in node.input_nodes]
                    node.output = node.forward(*node.inputs)
                else:
                    print("wait previous nodes to finish...")
                    continue

        return

def create_variable(node_name, node_params_: dict):
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
    op_cls = registers.get("operation", op_type, convert_name=False)
    operation = op_cls(**op_params)

    return operation


def run_session(config_filepath, feed_command=None, directory="./"):
    """Configure session with omegaconfig."""
    directory = pathlib.Path(directory)

    from omegaconf import OmegaConf

    # - add resolvers
    def create_vx_instance(vx_name, _root_):
        """"""
        vx_params = OmegaConf.to_object(_root_.variables.get(vx_name))

        return create_variable(vx_name, vx_params)

    OmegaConf.register_new_resolver(
        "gdp_v", create_vx_instance, use_cache=False
    )

    def create_op_instance(op_name, _root_):
        """"""
        op_params = OmegaConf.to_object(_root_.operations.get(op_name))

        return create_operation(op_name, op_params)

    OmegaConf.register_new_resolver(
        "gdp_o", create_op_instance, use_cache=False
    )

    # --
    def read_json(input_file):
        with open(input_file, "r") as fopen:
            input_dict = json.load(fopen)

        return input_dict

    OmegaConf.register_new_resolver(
        "json", read_json
    )

    # -- 
    from GDPy.builder.interface import build
    OmegaConf.register_new_resolver(
        "structure", lambda x: build(**{"builders":[create_variable("xxx", {"type": "builder", "method": "molecule", "filename": x})]})
    )


    # - configure
    conf = OmegaConf.load(config_filepath)

    # - add placeholders and their directories
    conf.placeholders = {}
    if feed_command is not None:
        pairs = [x.split("=") for x in feed_command]
        for k, v in pairs:
            conf.placeholders[k] = v
    #print("YAML: ", OmegaConf.to_yaml(conf))

    # - check operations and their directories
    for op_name, op_params in conf.operations.items():
        op_params["directory"] = str(directory/op_name)
    
    # - set variable directory
    for k, v_dict in conf.variables.items():
        v_dict["directory"] = str(directory/"variables"/k)
    print("YAML: ", OmegaConf.to_yaml(conf))

    # - resolve sessions
    container = OmegaConf.to_object(conf.sessions)
    for k, v in container.items():
        print(k, v)

    # - run session
    names = conf.placeholders.get("names", None)
    if names is not None:
        session_names = [x.strip() for x in names.strip().split(",")]
    else:
        session_names =[None]*len(container)

    exec_mode = conf.get("mode", "seq")
    if exec_mode == "seq":
        # -- sequential
        for i, (k, v) in enumerate(container.items()):
            n = session_names[i]
            if n is None:
                n = k
            entry_operation = v
            session = Session(directory=directory/n)
            session.run(entry_operation, feed_dict={})
    elif exec_mode == "cyc":
        # -- iterative
        session = CyclicSession(directory="./")
        session.run(
            container["init"], container["iter"], container.get("post"),
            repeat=conf.get("repeat", 1)
        )
    else:
        ...

    return


if __name__ == "__main__":
    ...
