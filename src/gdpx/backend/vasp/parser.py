import re

import numpy as np


def read_outcar_scf(outcar_lines: list[str]) -> dict:
    """"""
    vasp_params_from_outcar = {}

    for line in outcar_lines[:2000]:  # Only search the first 2000 lines
        if "ISPIN" in line:
            m = re.search(r"ISPIN\s*=\s*(\d+)", line)
            if m:
                vasp_params_from_outcar["ispin"] = int(m.group(1))

        if "NELM" in line:
            m = re.search(r"NELM\s*=\s*(\d+)", line)
            if m:
                vasp_params_from_outcar["nelm"] = int(m.group(1))

        if "EDIFF" in line:
            m = re.search(r"EDIFF\s*=\s*([0-9Ee\+\-\.]+)", line)
            if m:
                vasp_params_from_outcar["ediff"] = float(m.group(1))

        if all(k in vasp_params_from_outcar for k in ["ispin", "nelm", "ediff"]):
            break

    return vasp_params_from_outcar


def read_oszicar(lines: list[str], nelm: int, ediff: float) -> list[bool]:
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


def read_report(lines: list[str]):
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
