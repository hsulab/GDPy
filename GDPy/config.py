#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""Some shared configuration parameters.
"""

from typing import Union

#: number of parallel jobs for joblib
NJOBS: int = 1

# - find default vasp settings
#gdpconfig = Path.home() / ".gdp"
#if gdpconfig.exists() and gdpconfig.is_dir():
#    # find vasp config
#    vasprc = gdpconfig / "vasprc.json"
#    with open(vasprc, "r") as fopen:
#        input_dict = json.load(fopen)
#else:
#    input_dict = {}

if __name__ == "__main__":
    pass