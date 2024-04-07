#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import subprocess
from pathlib import Path
from itertools import groupby
from operator import itemgetter

from typing import Any, Union, List

import json
import yaml


def dict2str(d: dict, indent: int = 2):
    """Convert a nested dict to str."""

    def _dict2str(d_: dict, indent_: int):
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


class CustomTimer:

    def __init__(self, name="code", func=print):
        """"""
        self.name = name
        self._print = func

        return

    def __call__(self, func) -> Any:
        """"""

        def func_timer(*args, **kwargs):
            st = time.time()
            ret = func(*args, **kwargs)
            et = time.time()
            content = (
                "*** " + self.name + " time: " + "{:>8.4f}".format(et - st) + " ***"
            )
            self._print(content)

            return ret

        return func_timer

    def __enter__(self):
        """"""
        self.st = time.time()  # start time

        return self

    def __exit__(self, *args):
        """"""
        self.et = time.time()  # end time

        content = (
            "*** "
            + self.name
            + " time: "
            + "{:>8.4f}".format(self.et - self.st)
            + " ***"
        )
        self._print(content)

        return


def check_path(target_dir: Union[str, Path]) -> bool:
    """check path existence, if so skip the following
    TODO: add output option
    make this into a context?
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
    else:
        print(f"  {target_dir.name} exists, so next...")

    return


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


def convert_indices(indices: Union[str, List[int]], index_convention="lmp"):
    """parse indices for reading xyz by ase, get start for counting
    constrained indices followed by lammps convention
    "2:4 3:8"
    convert [1,2,3,6,7,8] to "1:3 6:8"
    lammps convention starts from 1 and includes end
    ---
    input can be either py or lmp
    output for indices is in py since it can be used to access atoms
    output for text is in lmp since it can be used in lammps or sth
    """
    ret = []
    if isinstance(indices, str):
        # string to List[int]
        for x in indices.strip().split():
            cur_range = list(map(int, x.split(":")))
            if len(cur_range) == 1:
                start, end = cur_range[0], cur_range[0]
            else:
                start, end = cur_range
            if index_convention == "lmp":
                ret.extend([i - 1 for i in list(range(start, end + 1))])
            elif index_convention == "py":
                ret.extend(list(range(start, end)))
            else:
                pass
    elif isinstance(indices, list):
        # List[int] to string
        indices = sorted(indices)
        if index_convention == "lmp":
            pass
        elif index_convention == "py":
            indices = [i + 1 for i in indices]
        ret = []
        # ranges = []
        for k, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
            group = map(itemgetter(1), g)
            group = list(map(int, group))
            # ranges.append((group[0],group[-1]))
            if group[0] == group[-1]:
                ret.append(str(group[0]))
            else:
                ret.append("{}:{}".format(group[0], group[-1]))
        ret = " ".join(ret)
    else:
        pass

    return ret


def parse_input_file(
    input_fpath: Union[str, Path],
    write_json: bool = False,  # write readin dict to check if alright
):
    """"""
    input_dict = None

    # - parse input type
    if isinstance(input_fpath, dict):
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
        for key, value in input_dict.items():
            key_dict = parse_input_file(value, write_json=False)
            if key_dict is not None:
                input_dict[key] = key_dict

    if input_dict and write_json:
        with open(json_path / "params.json", "w") as fopen:
            json.dump(input_dict, fopen, indent=4)
        print("See params.json for values of all parameters...")

    return input_dict


if __name__ == "__main__":
    ...
