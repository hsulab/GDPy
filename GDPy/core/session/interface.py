#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pathlib
import yaml

from GDPy import config
from .utils import create_variable, create_operation

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
        "vx", create_vx_instance, use_cache=False
    )

    def create_op_instance(op_name, _root_):
        """"""
        op_params = OmegaConf.to_object(_root_.operations.get(op_name))

        return create_operation(op_name, op_params)

    OmegaConf.register_new_resolver(
        "op", create_op_instance, use_cache=False
    )

    # --
    def read_json(input_file):
        with open(input_file, "r") as fopen:
            input_dict = json.load(fopen)

        return input_dict

    OmegaConf.register_new_resolver(
        "json", read_json
    )

    def read_yaml(input_file):
        with open(input_file, "r") as fopen:
            input_dict = yaml.safe_load(fopen)

        return input_dict

    OmegaConf.register_new_resolver(
        "yaml", read_yaml
    )

    # -- 
    from GDPy.builder.interface import build
    def load_structure(x):
        """"""
        builder = create_variable(
            "structure", {"type": "builder", "method": "direct", "frames": x}
        )
        node = build(builder)

        return node

    OmegaConf.register_new_resolver(
        "structure",  load_structure
    )


    # - configure
    conf = OmegaConf.load(config_filepath)

    # - add placeholders and their directories
    if "placeholders" not in conf:
        conf.placeholders = {}
    if feed_command is not None:
        pairs = [x.split("=") for x in feed_command]
        for k, v in pairs:
            conf.placeholders[k] = v
    config._debug(f"YAML: {OmegaConf.to_yaml(conf)}")

    # - check operations and their directories
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

    exec_mode = conf.get("mode", "seq")
    if exec_mode == "seq":
        from .session import Session
        # -- sequential
        for i, (k, v) in enumerate(container.items()):
            n = session_names[i]
            if n is None:
                n = k
            entry_operation = v
            session = Session(directory=directory/n)
            session.run(entry_operation, feed_dict={})
    elif exec_mode == "cyc":
        from .otf import CyclicSession
        # -- iterative
        session = CyclicSession(directory="./")
        session.run(
            container["init"], container["iter"], container.get("post"),
            repeat=conf.get("repeat", 1)
        )
    elif exec_mode == "otf":
        config._print("Use OTF Session...")
        from .otf import OTFSession
        for i, (k, v) in enumerate(container.items()):
            n = session_names[i]
            if n is None:
                n = k
            entry_operation = v
            session = OTFSession(directory=directory/n)
            session.run(entry_operation, feed_dict={})
    else:
        ...

    return

if __name__ == "__main__":
    ...
