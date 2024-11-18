#!/usr/bin/env python3
# -*- coding: utf-8 -*

""" potential wrappers
    general potential format: deepmd, eann, lasp, vasp
    dynamics backend: ase, lammps, lasp, vasp
"""


from .. import config
from ..core.register import registers

from ..computation.lammps import Lammps
from ..utils.logio import remove_extra_stream_handlers


if __name__ == "__main__":
    ...
