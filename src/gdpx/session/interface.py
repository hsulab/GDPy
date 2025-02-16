#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import json
import logging
import pathlib
import time
import traceback
from typing import Optional, Union

import omegaconf
import yaml
from omegaconf import OmegaConf

from gdpx import config
from .utils import create_operation, create_variable


class SessionInitialiser:

    #: Shared node objects.
    cache_nodes = {}

    @staticmethod
    def instantiate_operation(op_name, op_params):
        """Instantiate an operation."""
        params = {}
        for k, v in op_params.items():
            params[k] = v  # resolve one by one...
        op = create_operation(op_name, op_params)

        return op

    @staticmethod
    def resolve_operations(config: dict):
        """Initialise operations."""
        operations = {}
        for op_name, op_params in config.items():
            op = SessionInitialiser.instantiate_operation(op_name, op_params)
            operations[op_name] = op
            SessionInitialiser.cache_nodes[op_name] = op

        return operations

    @staticmethod
    def register_custom_resolvers():
        """Add some custom resolvers."""

        # Convert dictionary to object
        def create_vx_instance(vx_name, _root_):
            """"""
            if vx_name not in SessionInitialiser.cache_nodes:
                vx_params = OmegaConf.to_object(_root_.variables.get(vx_name))
                vx = create_variable(vx_name, vx_params)
                SessionInitialiser.cache_nodes[vx_name] = vx
                return vx
            else:
                return SessionInitialiser.cache_nodes[vx_name]

        OmegaConf.register_new_resolver(
            "vx", create_vx_instance, use_cache=False
        )

        def create_op_instance(op_name: str, _root_):
            """"""
            if op_name not in SessionInitialiser.cache_nodes:
                op_params = OmegaConf.to_object(_root_.operations.get(op_name))
                op = create_operation(op_name, op_params)
                SessionInitialiser.cache_nodes[op_name] = op
                return op
            else:
                return SessionInitialiser.cache_nodes[op_name]

        OmegaConf.register_new_resolver(
            "op", create_op_instance, use_cache=False
        )

        # Convert file to dictionary
        def read_json(input_file):
            with open(input_file, "r") as fopen:
                input_dict = json.load(fopen)

            return input_dict

        OmegaConf.register_new_resolver("json", read_json)

        def read_yaml(input_file):
            with open(input_file, "r") as fopen:
                input_dict = yaml.safe_load(fopen)

            return input_dict

        OmegaConf.register_new_resolver("yaml", read_yaml)

        return

    @staticmethod
    def convert_config_into_nodes(
        directory, config_dict: dict, feed_command: Optional[list] = None
    ):
        """"""
        # load configuration and resolve it
        conf = OmegaConf.create(config_dict)

        # add placeholders and their directories
        if "placeholders" not in conf:
            conf.placeholders = {}
        if feed_command is not None:
            pairs = [x.split("=") for x in feed_command]
            for k, v in pairs:
                if v.isdigit():
                    v = int(v)
                conf.placeholders[k] = v
        config._debug(f"YAML: {OmegaConf.to_yaml(conf)}")

        # check operations and their directories
        if "operations" not in conf:
            conf.operations = {}
        num_operations = len(conf.operations)
        if not (num_operations > 0):
            raise RuntimeError(f"No operations is found in the session.")
        for op_name, op_params in conf.operations.items():
            op_params["directory"] = str(directory / op_name)

        # set variable directory
        if "variables" not in conf:
            conf.variables = {}
        for k, v_dict in conf.variables.items():
            v_dict["directory"] = str(directory / "variables" / k)
        # print("YAML: ", OmegaConf.to_yaml(conf))

        # - resolve sessions
        # container = OmegaConf.to_object(conf.sessions)
        # for k, v in container.items():
        #    print(k, v)

        try:
            operations = SessionInitialiser.resolve_operations(
                conf["operations"]
            )
        except omegaconf.errors.InterpolationResolutionError as err:
            config._debug(traceback.format_exc())
            err_key = (str(err).strip().split("\n")[1]).strip().split(":")[1]
            config._print(f"FAILED TO PARSE `{err_key}` KEY.")
            err_info_tail = traceback.format_exc().split("\n")[-10:]
            for e in err_info_tail:
                config._print(f"{e}")
            exit()

        container = {}
        for k, v in conf["sessions"].items():
            container[k] = operations[v]

        # - run session
        names = conf.placeholders.get("names", None)
        if names is not None:
            session_names = [x.strip() for x in names.strip().split(",")]
        else:
            session_names = [None] * len(container)

        # get session general configs
        sconfigs = conf.get("configs", {})

        # some imported packages change `logging.basicConfig`
        # and accidently add a StreamHandler to logging.root
        # so remove it...
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                logging.root.removeHandler(h)

        return container, session_names, sconfigs


def run_session_once(
    config_dict: dict,
    feed_command: Optional[list[str]] = None,
    directory: Union[str, pathlib.Path] = "./",
):
    """Configure session with omegaconfig."""
    # set directory
    directory = pathlib.Path(directory)

    container, entry_nodes, session_config = (
        SessionInitialiser.convert_config_into_nodes(
            directory, config_dict, feed_command
        )
    )

    exec_mode = session_config.get("mode", "basic")
    if exec_mode == "basic":  # sequential
        from .basic import Session

        session_states = []
        for i, (k, v) in enumerate(container.items()):
            n = entry_nodes[i]
            if n is None:
                n = k
            entry_operation = v
            session = Session(directory=directory / n)
            session.run(entry_operation, feed_dict={})
            session_states.append(session.is_finished())
    elif exec_mode == "active":
        from .active import ActiveSession

        assert len(container) == 1, "ActiveSession only accepts one operation."

        session_states = []
        for i, (k, v) in enumerate(container.items()):
            n = entry_nodes[i]
            if n is None:
                n = k
            entry_operation = v
            session = ActiveSession(
                steps=session_config.get("steps", 2),
                reset_random_state=session_config.get(
                    "reset_random_state", False
                ),
                reset_random_config=session_config.get(
                    "reset_random_config", ("init", 0)
                ),
                directory=directory / n,
            )
            session.run(entry_operation, feed_dict={})
            # config._print(f"{session.state =}")
            session_states.append(session.is_finished())
            # config._print(f"{session.state =}")
    else:
        session_states = [False]
        raise RuntimeError(f"Unknown session type {exec_mode}.")

    return all(session_states)


def run_session(
    config_filepath: Union[str, pathlib.Path],
    feed_command: Optional[list[str]] = None,
    timewait: float = -1.0,
    directory: Union[str, pathlib.Path] = "./",
):
    """Configure session with omegaconfig."""
    # Check working directory and input file.
    directory = pathlib.Path(directory)

    config_filepath = pathlib.Path(config_filepath)

    if config_filepath.suffix == ".json":
        with open(config_filepath, "r") as fopen:
            config_dict = json.load(fopen)
    elif config_filepath.suffix == ".yaml":
        with open(config_filepath, "r") as fopen:
            config_dict = yaml.safe_load(fopen)
    else:
        raise RuntimeError(f"Fail to load config `{str(config_filepath)}`")

    raw_config_dict = config_dict

    SessionInitialiser.register_custom_resolvers()

    # Run session repeatedly.
    # We may not use an explicit daemon here as it may be killed by the
    # administrator.
    if timewait > 0:
        for i in range(1000):
            SessionInitialiser.cache_nodes = (
                {}
            )  # Clear cache before a new run.
            config._print(f"... Daemon is running step {i:>04d} ...")
            config_dict = copy.deepcopy(raw_config_dict)
            is_finished = run_session_once(
                config_dict, feed_command, directory
            )
            if is_finished:
                break
            else:
                config._print(
                    f"... Daemon will sleep for {timewait} seconds ..."
                )
                time.sleep(timewait)
        else:
            config._print("Reach maximum monitor for-loop.")
    else:
        run_session_once(config_dict, feed_command, directory)

    return


if __name__ == "__main__":
    ...
