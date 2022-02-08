#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ast import parse
from pathlib import Path
import subprocess

from typing import Union

import json
import yaml

def run_command(directory, command, comment=''):
    proc = subprocess.Popen(
        command, shell=True, cwd=directory, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding = "utf-8"
    )
    errorcode = proc.wait() # 10 seconds
    msg = "Message: " + "".join(proc.stdout.readlines())
    print(msg)
    if errorcode:
        raise ValueError('Error in %s at %s.' %(comment, directory))
    
    return msg

def parse_input_file(
    input_fpath: Union[str,Path],
    write_json: bool = False # write readin dict to check if alright
):
    """"""
    input_dict = None
    
    input_file = Path(input_fpath)
    if input_file.suffix == ".json":
        with open(input_file, "r") as fopen:
            input_dict = json.load(fopen)
    elif input_file.suffix == ".yaml":
        with open(input_file, "r") as fopen:
            input_dict = yaml.safe_load(fopen)
    else:
        pass
        # raise ValueError("input file format should be json or yaml...")

    if input_dict and write_json:
        with open(input_file.parent/"params.json", "w") as fopen:
            json.dump(input_dict, fopen, indent=4)
        print("See params.json for values of all parameters...")

    return input_dict
