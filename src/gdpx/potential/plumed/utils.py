#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
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


def update_plumed_input_lines_by_driver(
    input_lines: list[str], wdir: str, stride: int, temperature: float
) -> list[str]:
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


if __name__ == "__main__":
    ...
