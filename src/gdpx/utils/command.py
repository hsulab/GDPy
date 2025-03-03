#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import pathlib
from typing import Any

import yaml


def parse_input_file(
    input_fpath: Any,
    write_json: bool = False,  # write readin dict to check if alright
) -> dict:
    """"""
    input_dict = None

    # Check input type
    if isinstance(input_fpath, list):
        input_dict = input_fpath
        json_path = pathlib.Path.cwd()
    elif isinstance(input_fpath, dict):
        input_dict = input_fpath
        json_path = pathlib.Path.cwd()
    else:
        if isinstance(input_fpath, str):
            input_file = pathlib.Path(input_fpath)
            json_path = input_file.parent
        elif isinstance(input_fpath, pathlib.Path):
            input_file = input_fpath
            json_path = input_file.parent
        else:
            return None

        # Load dict from a file
        try:
            if input_file.suffix == ".json":
                with open(input_file, "r") as fopen:
                    input_dict = json.load(fopen)
            elif input_file.suffix == ".yaml":
                with open(input_file, "r") as fopen:
                    input_dict = yaml.safe_load(fopen)
            else:
                ...
        except FileNotFoundError:
            # There is json or yaml in the string but it is not a file though.
            input_dict = None

    # Recursively read internal json or yaml files
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
