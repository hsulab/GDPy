#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple

import numpy as np


def parse_thermo_data(lines, print_func=print, debug_func=print) -> Tuple[dict, str]:
    """Read energy ... results from log.lammps file."""
    # parse input lines
    idx, start_idx, end_idx = None, None, None
    for idx, line in enumerate(lines):
        # get the line index at the start of the thermo infomation
        # test with 29Oct2020 and 23Jun2022
        if line.strip().startswith("Step"):
            start_idx = idx
        # NOTE: find line index at the end
        if line.strip().startswith("ERROR: "):  # Lost atoms
            end_idx = idx
        if line.strip().startswith("Loop time"):
            end_idx = idx
        if start_idx is not None and end_idx is not None:
            break
    else:
        end_idx = idx
    debug_func(f"INITIAL LAMMPS LOG INDEX: {start_idx} {end_idx}")
    assert (
        start_idx is not None and end_idx is not None
    ), f"INITIAL LAMMPS LOG INDEX: {start_idx} {end_idx}"

    # check valid lines
    end_info = ""
    # sometimes the line may not be complete
    ncols = len(lines[start_idx].strip().split())
    for i in range(end_idx, start_idx, -1):
        curr_data = lines[i].strip().split()
        curr_ncols = len(curr_data)
        debug_func(f"  LAMMPS LINE: {lines[i].strip()}")
        if curr_ncols == ncols:  # The Line has the full thermo info...
            try:
                step = int(curr_data[0])
                end_idx = i + 1
            except ValueError:
                ...
            finally:
                end_info = lines[i].strip()
                debug_func(f"  LAMMPS STEP: {end_info}")
                break
        else:
            ...
    else:
        end_idx = None  # even not one single complete line
    debug_func(f"FINAL   LAMMPS LOG INDEX: {start_idx} {end_idx}")

    if start_idx is None or end_idx is None:
        raise RuntimeError(f"ERROR   LAMMPS LOG INDEX {start_idx} {end_idx}.")

    # -- parse index of PotEng
    # TODO: save timestep info?
    thermo_keywords = lines[start_idx].strip().split()
    if "PotEng" not in thermo_keywords:
        raise RuntimeError(f"Cannot find PotEng in lammps output.")
    thermo_data = []
    for x in lines[start_idx + 1 : end_idx]:
        x_data = x.strip().split()
        if x_data[0].isdigit():  # There may have some extra warnings... such as restart
            thermo_data.append(x_data)
    # thermo_data = np.array([line.strip().split() for line in thermo_data], dtype=float).transpose()
    thermo_data = np.array(thermo_data, dtype=float).transpose()
    thermo_dict = {}
    for i, k in enumerate(thermo_keywords):
        thermo_dict[k] = thermo_data[i]

    return thermo_dict, end_info


if __name__ == "__main__":
    ...
