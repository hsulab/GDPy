import copy
import json
import pathlib
import time
from typing import Optional, Union

import yaml

from gdpx import config
from gdpx.session.interface import SessionInitialiser, run_session_from_dict


def run_session(
    config_filepath: Union[str, pathlib.Path],
    feed_command: Optional[list[str]] = None,
    timewait: float = -1.0,
    timemax: float = -1.0,
    num_repeats: int = 1000,
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
    st = time.time()
    if timewait > 0:
        for i in range(num_repeats):
            SessionInitialiser.cache_nodes = {}  # Clear cache before a new run.
            config._print("\x1b[1;32;40m" + f"... Daemon is running step {i:>04d} ..." + "\x1b[0m")
            config_dict = copy.deepcopy(raw_config_dict)
            is_finished = run_session_from_dict(config_dict, feed_command, directory)
            if is_finished:
                break
            else:
                ct = time.time()
                if timemax > 0 and (ct - st + timewait) > timemax:
                    config._print("session reached the maxmum time.")
                    break

                config._print(f"... Daemon will sleep for {timewait} seconds ...")
                time.sleep(timewait)
        else:
            config._print("session reached the maximum repeats.")
    else:
        run_session_from_dict(config_dict, feed_command, directory)
    et = time.time()
    config._print(f"session time: {et - st:>.4f}s")

    return
