import copy
import pathlib
from typing import Callable


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


def update_plumed_input_lines_by_driver(input_lines: list[str], stride: int, temperature: float) -> list[str]:
    """Update the input lines with the some parameters from the driver setting."""
    input_lines, parsed_lines = copy.deepcopy(input_lines), []
    for line in input_lines:
        # parsed_line = update_input_value(line, "FILE", wdir, func=lambda x, y: os.path.join(y, x))
        parsed_line = update_input_value(line, "STRIDE", stride, func=lambda x, y: str(y))
        # Some parameters in metadynamics
        parsed_line = update_input_value(parsed_line, "PACE", stride, func=lambda x, y: str(y))
        parsed_line = update_input_value(parsed_line, "TEMP", temperature, func=lambda x, y: str(y))
        parsed_lines.append(parsed_line)

    return parsed_lines


def write_plumed_input_file(plumed_inp_fpath: str, input_lines: list[str], driver_params: dict) -> None:
    """Write the plumed input file."""
    # We must have those parameters from the host driver
    dump_period = driver_params.get("dump_period")
    assert isinstance(dump_period, int), f"dump_period must be an integer instead of {type(dump_period)}."

    temperature = driver_params.get("temperature")
    assert isinstance(temperature, int) or isinstance(temperature, float), (
        f"temperature must be a float or an int instead of {type(temperature)}."
    )

    # TODO: We need constrain the FILE to be HILLS and COLVAR in PRINT and METAD,
    #       and well-tempered metad should not be in annealing.
    plumed_inp_lines = update_plumed_input_lines_by_driver(
        input_lines,
        stride=dump_period,
        temperature=temperature,
    )
    with open(plumed_inp_fpath, "w") as fopen:
        fopen.write("".join(plumed_inp_lines))

    return
