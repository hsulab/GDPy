#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""Some shared configuration parameters.
"""

import logging
from typing import Union, List, Callable

#: 
logger = logging.getLogger("GDP")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y%b%d-%H:%M:%S"
    #"%(levelname)s: %(module)s - %(message)s"
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

_print: Callable = logger.info
_debug: Callable = logger.debug

LOGO_LINES = [
"  ____ ____  ______  __ ",
" / ___|  _ \|  _ \ \/ / ",
"| |  _| | | | |_) \  /  ",
"| |_| | |_| |  __//  \  ",
" \____|____/|_|  /_/\_\ ",
"                        ",
]

#: Number of parallel jobs for joblib.
NJOBS: int = 1

#: Global random number generator
GRNG = None

# - find default vasp settings
#gdpconfig = Path.home() / ".gdp"
#if gdpconfig.exists() and gdpconfig.is_dir():
#    # find vasp config
#    vasprc = gdpconfig / "vasprc.json"
#    with open(vasprc, "r") as fopen:
#        input_dict = json.load(fopen)
#else:
#    input_dict = {}

#: Model deviations by the committee model.
VALID_DEVI_FRAME_KEYS: List[str] = [
    "devi_te",
    "max_devi_v", "min_devi_v", "avg_devi_v",
    "max_devi_f", "min_devi_f", "avg_devi_f",
    "max_devi_ae", "min_devi_ae", "avg_devi_ae",
]

#: Model deviations by the committee model.
VALID_DEVI_ATOMIC_KEYS: List[str] = [
    "devi_f",
]

if __name__ == "__main__":
    ...