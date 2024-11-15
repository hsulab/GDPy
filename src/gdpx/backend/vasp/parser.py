#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
from typing import List, Tuple

import numpy as np


def read_outcar_scf(lines: List[str]) -> Tuple[int, float]:
    """"""
    nelm, ediff = None, None
    for line in lines:
        if line.strip().startswith("NELM"):
            nelm = int(line.split()[2][:-1])
        if line.strip().startswith("EDIFF"):
            ediff = float(line.split()[2])
        if nelm is not None and ediff is not None:
            break
    else:
        ...  # TODO: raise an error?

    return nelm, ediff

def read_oszicar(lines: List[str], nelm: int, ediff: float) -> List[bool]:
    """"""
    convergence = []
    content = ""
    for line in lines:
        start = line.strip().split()[0]
        if start == "N":
            content = ""
            continue
        if start.isdigit():
            scfsteps = [int(s.split()[1]) for s in content.strip().split("\n")]
            num_scfsteps = len(scfsteps)
            assert num_scfsteps == scfsteps[-1], f"{num_scfsteps =}, {scfsteps[-1] =}"
            enediffs = [float(s.split()[3]) for s in content.strip().split("\n")]
            is_converged = num_scfsteps < nelm or np.fabs(enediffs[-1]) <= ediff
            convergence.append(is_converged)
        content += line

    # If the last SCF is not finished,
    # there is no content to check

    return convergence


def read_report(lines: List[str]):
    """Read VASP-REPORT and find RANDOM_SEED."""
    pattern = re.compile(r"RANDOM_SEED =\s*(\d+)\s+(\d+)\s+(\d+)")
    random_seeds = []
    for line in lines:
        match = pattern.search(line)
        if match:
            random_seeds.append([match.group(1), match.group(2), match.group(3)])
        else:
            ...
    random_seeds = np.array(random_seeds, dtype=np.int32)

    return random_seeds


if __name__ == "__main__":
    ...
  
