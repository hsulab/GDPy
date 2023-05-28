#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
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

    def __init__(self, init, iteration, post=None, directory="./") -> None:
        """"""
        self.directory = pathlib.Path(directory)

        self.init = init
        self.loop = iteration
        self.post = post

        return
    
    def iteration(self, operations):
        """Iterative part that converges at certain criteria.

        This noramlly includes steps: sample, select, label, and train. Some inputs of
        operations should be update during the iterations, for instance, the models 
        in the potential.

        """

        return
    
    def run(self, feed_dict={}, *args, **kwargs) -> NoReturn:
        """"""
        # - init
        session, end_node, placeholders = self.init
        self._debug("init session: ", session)
        self._debug("entry point: ", end_node)
        self._debug("placeholders: ", placeholders)

        self._print("[{:^24s}]".format("INIT"))
        session.run(end_node, )

        # - iter
        self._print("[{:^24s}]".format("ITER"))
        for i in range(5):
            session, end_node, placeholders = self.loop
            session.directory = self.directory/"iter"/f"iter.{str(i).zfill(4)}"
            # TODO: update some parameters
            session.run(end_node, )
            # TODO: check if the last node is finished
            ...

        # - post
        if self.post is not None:
            ...

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
    #op_method = op_params.pop("method", None)
    #op_func = registers.get("operation", op_type, convert_name=False)
    #return op_func, op_params

    op_cls = registers.get("operation", op_type, convert_name=False)
    operation = op_cls(**op_params)

    return operation

def create_op_instance(op_name, *args, **kwargs):
    """"""
    op_type = kwargs.pop("type", None)
    assert op_type is not None, f"{op_name} has no type."

    operation = registers.create(
        "operation", op_type, convert_name=False, **kwargs
    )

    return operation

def create_session(session_params, phs_params, nodes_params, ops_params, temp_nodes, directory="./", label=None):
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

def naive_run_session(config_filepath, custom_session_names=None, entry_string: str=None, directory="./", label=None):
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

    phs_params = session_config.get("placeholders", {}) # this is optional
    nodes_params = session_config.get("nodes", None)
    ops_params = session_config.get("operations", None)
    sessions_params = session_config.get("sessions", None)

    # - try use session label
    nsessions = len(sessions_params)
    if label is not None:
        assert nsessions == 1, f"Label can be used for only one session."

    # - create sessions
    temp_nodes = {} # intermediate nodes, shared among sessions

    sessions = {}
    for name, cur_params in sessions_params.items():
        if label is not None:
            name = label
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

@registers.operation.register
class enter(Operation):

    def __init__(self, directory="./", *args, **kwargs) -> NoReturn:
        """"""
        input_nodes = list(kwargs.values())
        super().__init__(input_nodes=input_nodes, directory=directory)

        return
    
    def forward(self, *args):
        super().forward()

        return


def run_session(config_filepath, feed_command=None, custom_session_names=None, entry_string: str=None, directory="./", label=None):
    """Configure session with omegaconfig."""
    directory = pathlib.Path(directory)

    from omegaconf import OmegaConf

    # - add resolvers
    def create_vx_instance(vx_name, _root_):
        """"""
        #pattern = "${variables.%s}" %vx_name
        #print("pattern: ", pattern)
        vx_params = OmegaConf.to_object(_root_.variables.get(vx_name))

        return create_node(vx_name, vx_params)

    OmegaConf.register_new_resolver(
        #"gdp_v", lambda x: create_node("xxx", OmegaConf.to_object(x)),
        #use_cache=True
        "gdp_v", create_vx_instance, use_cache=False
    )

    def create_op_instance(op_name, _root_):
        """"""
        op_params = OmegaConf.to_object(_root_.operations.get(op_name))

        return create_operation(op_name, op_params)

    OmegaConf.register_new_resolver(
        #"gdp_o", lambda x: create_op_instance("xxx", **OmegaConf.to_object(x)),
        #use_cache=True
        "gdp_o", create_op_instance, use_cache=False
    )

    from GDPy.builder.interface import build
    OmegaConf.register_new_resolver(
        "structure", lambda x: build(**{"builders":[create_node("xxx", {"type": "builder", "method": "molecule", "filename": x})]})
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

    # -- check if every session has a valid entry operation
    #entry_dict = {}
    #for k, op_dict in conf.sessions.items():
    #    print("session: ", k)
    #    entry_names = []
    #    for op_name, op_params in op_dict.items():
    #        op_type = op_params.get("type")
    #        if op_type == "enter":
    #            entry_names.append(op_name)
    #        op_params["directory"] = str(directory/k/op_name)
    #        
    #    assert len(entry_names) == 1, f"Session {k} only needs one entry operation!!!"
    #    entry_dict[k] = entry_names[0]
    #    print(f"find entry: {entry_names[0]}")
    
    for op_name, op_params in conf.operations.items():
        op_params["directory"] = str(directory/op_name)
    
    # - set variable directory
    for k, v_dict in conf.variables.items():
        v_dict["directory"] = str(directory/"variables"/k)
    #print("YAML: ", OmegaConf.to_yaml(conf))

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

    for i, (k, v) in enumerate(container.items()):
        n = session_names[i]
        if n is None:
            n = k
        entry_operation = v
        session = Session(directory=directory/n)
        session.run(entry_operation, feed_dict={})

    return


if __name__ == "__main__":
    ...
