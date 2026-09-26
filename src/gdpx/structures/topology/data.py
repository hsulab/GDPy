from typing import NamedTuple

import numpy as np


class NeighbourData(NamedTuple):
    senders: np.ndarray
    receivers: np.ndarray
    distances: np.ndarray
    shifts: np.ndarray
