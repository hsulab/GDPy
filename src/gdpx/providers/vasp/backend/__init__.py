#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .parser import (
    read_report, read_oszicar, read_outcar_scf
)
from .writer import write_vasp

__all__ = [
    "read_report", "read_oszicar", "read_outcar_scf",
    "write_vasp"
]


if __name__ == "__main__":
    ...
  
