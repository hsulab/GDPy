#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Union

import yaml


def dictionary_to_string(d: Mapping[str, Any], indent: int = 2):
    """Convert a nested dict to str."""

    def _dict2str(d_: Mapping[str, Any], indent_: int):
        """Recursive function."""
        content = ""
        for k, v in d_.items():
            if isinstance(v, dict):
                content += f"{k}:\n" + _dict2str(v, indent_ + indent)
            else:
                content += " " * indent_ + f"{k}: {v}\n"

        return content

    content = _dict2str(d, 0)

    return content


def find_backups(dpath, fname, prefix="bak"):
    """find a series of files in a dir
    such as fname, bak.0.fname, bak.1.fname
    """
    dpath = Path(dpath)
    fpath = dpath / fname
    if not fpath.exists():
        raise FileNotFoundError(f"fpath does not exist.")

    backups = list(dpath.glob(prefix + ".[0-9]*." + fname))
    backups = sorted(backups, key=lambda x: int(x.name.split(".")[1]))
    backups.append(fpath)

    return backups


def run_command(directory, command, comment="", timeout=None):
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if timeout is None:
        errorcode = proc.wait()
    else:
        errorcode = proc.wait(timeout=timeout)

    msg = "Message: " + "".join(proc.stdout.readlines())
    print(msg)
    if errorcode:
        raise RuntimeError("Error in %s at %s." % (comment, directory))

    return msg


def parse_input_file(
    input_fpath: Union[str, Path],
    write_json: bool = False,  # write readin dict to check if alright
) -> dict:
    """"""
    input_dict = None

    # - parse input type
    if isinstance(input_fpath, list):
        input_dict = input_fpath
        json_path = Path.cwd()
    elif isinstance(input_fpath, dict):
        input_dict = input_fpath
        json_path = Path.cwd()
    else:
        if isinstance(input_fpath, str):
            input_file = Path(input_fpath)
            json_path = input_file.parent
        elif isinstance(input_fpath, Path):
            input_file = input_fpath
            json_path = input_file.parent
        else:
            return None

        # --- read dict from files
        try:
            if input_file.suffix == ".json":
                with open(input_file, "r") as fopen:
                    input_dict = json.load(fopen)
            elif input_file.suffix == ".yaml":
                with open(input_file, "r") as fopen:
                    input_dict = yaml.safe_load(fopen)
            else:
                ...
        except FileNotFoundError as e:
            # NOTE: There is json or yaml in the string but it is not a file though.
            input_dict = None

    # NOTE: recursive read internal json or yaml files
    if input_dict is not None:
        if isinstance(input_dict, dict):
            for key, value in input_dict.items():
                key_dict = parse_input_file(value, write_json=False)
                if key_dict is not None:
                    input_dict[key] = key_dict
        elif isinstance(input_dict, list):
            for i, data in enumerate(input_dict):
                new_data = parse_input_file(data, write_json=False)
                if new_data is not None:
                    input_dict[i] = new_data
        else:
            raise RuntimeError(f"Unknown input `{input_dict}`.")

    if input_dict and write_json:
        with open(json_path / "params.json", "w") as fopen:
            json.dump(input_dict, fopen, indent=4)
        print("See params.json for values of all parameters...")

    return input_dict


if __name__ == "__main__":
    ...
