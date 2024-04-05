#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
import pathlib

from ase.io import read, write

from gdpx.core.session.basic import Session


def run_cyclic_session_xxx(
    config_filepath,
    custom_session_names=None,
    entry_string: str = None,
    directory="./",
    label=None,
):
    """"""
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
                curr_data = read(data[s + 1 : e][0], ":")
            entry_data[curr_name] = curr_data
    print("entry", entry_data)

    # - parse nodes
    from GDPy.utils.command import parse_input_file

    session_config = parse_input_file(config_filepath)

    phs_params = session_config.get("placeholders", {})  # this is optional
    nodes_params = session_config.get("nodes", None)
    ops_params = session_config.get("operations", None)
    sessions_params = session_config.get("sessions", None)

    # - try use session label
    nsessions = len(sessions_params)
    if label is not None:
        assert nsessions == 1, f"Label can be used for only one session."

    # - create sessions
    temp_nodes = {}  # intermediate nodes, shared among sessions

    sessions = {}
    for name, cur_params in sessions_params.items():
        if label is not None:
            name = label
        sessions[name] = create_session(
            cur_params,
            phs_params,
            nodes_params,
            ops_params,
            temp_nodes,
            directory=directory / name,
        )

    cyclic_session = CyclicSession(init=sessions["init"], iteration=sessions["iter"])
    cyclic_session.run()

    # - run session
    # if custom_session_names is None:
    #    custom_session_names = copy.deepcopy(list(sessions.keys()))
    # for name, (session, end_node, placeholders) in sessions.items():
    #    feed_dict = {p:entry_data[p.name] for p in placeholders}
    #    if name in custom_session_names:
    #        print(f"===== run session {name} =====")
    #        _ = session.run(end_node, feed_dict=feed_dict)

    return


if __name__ == "__main__":
    import_all_modules_for_register()
    config_filepath = "/mnt/scratch2/users/40247882/porous/devel/pert.yaml"
    run_cyclic_session(config_filepath)
    ...
