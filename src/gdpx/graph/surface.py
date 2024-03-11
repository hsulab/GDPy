#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ase import Atoms
from ase.constraints import constrained_indices


"""Some surface-detection methods."""


def normalize(vector):
    """"""
    return vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else vector * 0


def generate_normals(
    atoms: Atoms, nl,
    surf_indices: List[int] = None,
    ads_indices: List[int] = [],
    mask_elements: List[str] = [],
    surf_norm_min: float = 0.5,
    normalised: bool=True, system: str = "surface"
):
    """Find normals to surface of a structure.

    Atoms that have positive z-axis normal would be on the surface.

    Args:
        atoms: Atoms object.
        nl: Pre-created neighbour list.
        surf_indices: Indices of surface atoms.
        ads_indices: Indices of adsorbate atoms.
        mask_elements: Indices of atoms that are not considered for adsorption sites.
        surf_norm_min: Minimum for a surface normal.
        normalised: Whether normalise normals.

    """
    natoms = len(atoms)

    normals = np.zeros(shape=(natoms, 3), dtype=float)
    for i, atom in enumerate(atoms):
        # print("centre: ", index, atom.position)
        if i in ads_indices:
            continue
        normal = np.array([0, 0, 0], dtype=float)
        for nei_idx, offset in zip(*nl.get_neighbors(i)):
            if nei_idx in ads_indices:
                continue
            # print("neigh: ", neighbor)
            direction = atom.position - (
                atoms[nei_idx].position + np.dot(offset, atoms.get_cell())
            )
            normal += direction
        # print("direction:", normal)
        if np.linalg.norm(normal) > surf_norm_min:
            normals[i, :] = normalize(normal) if normalised else normal

    # NOTE: check normal is pointed to z-axis surface
    if surf_indices is not None:
        surf_indices_ = surf_indices
        for i in range(natoms):
            if i not in surf_indices_:
                normals[i] = np.zeros(3)
    else:
        if system.startswith("surface"):
            surf_indices_ = [
                i
                for i in range(natoms)
                if np.linalg.norm(normals[i]) > 1e-5 and normals[i][2] > 0.0
            ]
        elif system.startswith("cluster"):
            surf_indices_ = [
                i for i in range(natoms) if np.linalg.norm(normals[i]) > 1e-5
            ]
        else:
            raise RuntimeError(f"Surface detection is not available for system {system}.")

    # - remove constrained atoms from surface atoms
    constrained = constrained_indices(atoms)
    surf_indices_ = [i for i in surf_indices_ if i not in constrained]

    # - remove unallowed elements
    surf_indices_ = [i for i in surf_indices_ if atoms[i].symbol not in mask_elements]

    return normals, surf_indices_


if __name__ == "__main__":
    ...