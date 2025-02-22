#!/usr/bin/env python3
# -*- coding: utf-8 -*


import logging
from typing import Callable

logger = logging.getLogger("GDP")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y%b%d-%H:%M:%S",
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

_print: Callable = logger.info
_debug: Callable = logger.debug

LOGO_LINES = [
    r"  ____ ____  ______  __ ",
    r" / ___|  _ \|  _ \ \/ / ",
    r"| |  _| | | | |_) \  /  ",
    r"| |_| | |_| |  __//  \  ",
    r" \____|____/|_|  /_/\_\ ",
    r"                        ",
]

#: Number of parallel jobs for joblib.
NJOBS: int = 1

#: Global random number generator
GRNG = None

#: Model deviations by the committee model.
VALID_DEVI_FRAME_KEYS: list[str] = [
    "devi_te",
    "max_devi_v",
    "min_devi_v",
    "avg_devi_v",
    "max_devi_f",
    "min_devi_f",
    "avg_devi_f",
    "max_devi_ae",
    "min_devi_ae",
    "avg_devi_ae",
]

#: Model deviations by the committee model.
VALID_DEVI_ATOMIC_KEYS: list[str] = [
    "devi_f",
]

if __name__ == "__main__":
    ...
