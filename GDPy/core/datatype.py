#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from ase import Atoms

#_AtomsFrames = type(List[Atoms])

def isAtomsFrames(obj) -> bool:
    """Check if the object is a List of Atoms objects."""

    return bool(obj) and all(isinstance(x, Atoms) for x in obj)

def isTrajectories(obj) -> bool:
    """Check if the object is a List of AtomsFrames objects."""

    return bool(obj) and all(isAtomsFrames(x) for x in obj)


if __name__ == "__main__":
    ...