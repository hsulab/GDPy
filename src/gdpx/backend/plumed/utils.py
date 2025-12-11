import copy
import pathlib
from typing import Callable

import numpy as np


def update_input_value(line: str, key: str, value, func: Callable[[str, str], str]) -> str:
    """Update the given key with the new value."""
    shift = len(key) + 1  # key name and =
    if line.find(key) != -1:
        ini = line.find(key)
        end = line.find(" ", ini)
        if end == -1:
            prev = line[ini + shift :]
            line = line[: ini + shift] + func(prev, value)
        else:
            prev = line[ini + shift : end]
            line = line[: ini + shift] + func(prev, value) + line[end:]
    if not line.endswith("\n"):
        line += "\n"

    return line


def update_plumed_input_lines_by_driver(
    input_lines: list[str], stride: int, ckpt_stride: int, temperature: float
) -> list[str]:
    """Update the input lines with the some parameters from the driver setting."""
    input_lines, parsed_lines = copy.deepcopy(input_lines), []
    for line in input_lines:
        parsed_line = update_input_value(line, "STRIDE", stride, func=lambda _, y: str(y))
        # Some parameters in metadynamics
        parsed_line = update_input_value(parsed_line, "PACE", stride, func=lambda _, y: str(y))
        parsed_line = update_input_value(parsed_line, "TEMP", temperature, func=lambda _, y: str(y))
        # For opes,
        parsed_line = update_input_value(parsed_line, "STATE_WSTRIDE", ckpt_stride, func=lambda _, y: str(y))
        parsed_lines.append(parsed_line)

    return parsed_lines


def write_plumed_input_file(
    working_directory: pathlib.Path, input_lines: list[str], driver_params: dict, is_continue: bool = False
) -> None:
    """Write the plumed input file."""
    # We must have those parameters from the host driver
    dump_period = driver_params.get("dump_period")
    assert isinstance(dump_period, int), f"dump_period must be an integer instead of {type(dump_period)}."

    ckpt_period = driver_params.get("ckpt_period")
    assert isinstance(ckpt_period, int), f"ckpt_period must be an integer instead of {type(ckpt_period)}."

    temperature = driver_params.get("temperature")
    assert isinstance(temperature, int) or isinstance(temperature, float), (
        f"temperature must be a float or an int instead of {type(temperature)}."
    )

    # TODO: We need constrain the FILE to be HILLS and COLVAR in PRINT and METAD,
    #       and well-tempered metad should not be in annealing.
    plumed_inp_lines = update_plumed_input_lines_by_driver(
        input_lines,
        stride=dump_period,
        ckpt_stride=ckpt_period,
        temperature=temperature,
    )

    # Add RESTART line if necessary
    if is_continue:
        # Add global restart line
        restart_line = "RESTART\n"
        plumed_inp_lines.insert(0, restart_line)
        # Check if we have OPES_METAD, if so, we need to add STORE_STATES if not present
        line_index_opes_metad = [i for i, line in enumerate(plumed_inp_lines) if "OPES_METAD" in line]
        if line_index_opes_metad:
            line_index = line_index_opes_metad[0]
            input_arguments = plumed_inp_lines[line_index].split()
            extra_arguments = ""
            if "STORE_STATES" not in input_arguments:
                extra_arguments = "STORE_STATES"
            if "STATE_RFILE" not in input_arguments:
                # check if we do have STATE, sometimes it is broken and not copied from the previous simulation
                if (working_directory / "STATE").is_file():
                    extra_arguments += " STATE_RFILE=STATE"
            # extra_arguments += " RESTART"
            plumed_inp_lines[line_index] = plumed_inp_lines[line_index].strip() + " " + extra_arguments + "\n"

    with open(working_directory / "plumed.inp", "w") as fopen:
        fopen.write("".join(plumed_inp_lines))

    return


def clap_plumed_file_by_number(
    prev_fpath: pathlib.Path, curr_fpath: pathlib.Path, num_steps: int, offset: int = 0
) -> None:
    """Clap COLVAR or HILLS by the number of steps.

    Note:
        HILLS does not dump the first step (step 0).

    Args:
        prev_fpath: The previous COLVAR or HILLS file path.
        curr_fpath: The current COLVAR or HILLS file path to be written.
        num_steps: The number of steps to clap to.
        offset: The offset number of lines to add (0 for COLVAR and HILLS).

    Returns:
        None

    """
    # Check if there are multiple comment lines due to multiple restarts
    with open(prev_fpath, "r") as fopen:
        lines = fopen.readlines()
    comment_lines = [i for i, line in enumerate(lines) if line.startswith("#")]
    num_comments = len(comment_lines)  # COLVAR has 1 per restart, HILLS has 3

    # TODO: If there is no outputs after the latest restart?

    num_lines_needed = num_steps + num_comments + offset

    # Check if the previous COLVAR has enough lines
    num_lines = len(lines)
    assert num_lines >= num_lines_needed, (
        f"The previous COLVAR file {prev_fpath} has {num_lines} lines and the last line is `{lines[-1].strip()}`, "
        f"which is less than the required {num_lines_needed} lines by step {num_steps}."
    )

    with open(curr_fpath, "w") as fopen:
        fopen.writelines(lines[:num_lines_needed])

    return


def clap_plumed_file_by_simulations(prev_fpath: pathlib.Path, curr_fpath: pathlib.Path, finished_time: float):
    """Split STATE by simulations and take the last one."""
    with open(prev_fpath, "r") as fopen:
        lines = fopen.readlines()

    # Find all the indices start with '#! FIELDS time'
    field_lines = [i for i, line in enumerate(lines) if line.startswith("#! FIELDS time")]
    num_simulations = len(field_lines)
    if num_simulations > 0:
        last_sim_lines = lines[field_lines[-1] :]
        last_line = last_sim_lines[-1]
        time_in_state = int(last_line.strip().split()[0])
        if np.fabs(time_in_state - finished_time) < 1e-4:  # should be equal
            with open(curr_fpath, "w") as fopen:
                fopen.writelines(last_sim_lines)
        else:
            ...  # skip state write if not match
    else:
        ...  # skip state write if no simulations found

    return
