#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import functools
import itertools
from typing import Callable, Optional

import numpy as np
from ase import Atoms

from gdpx.utils.atoms_tags import reassign_tags_by_species

from .particle import translate_then_rotate
from .spatial import check_atomic_distances


def prepare_monodentate_adsorbate(site, adsorbate: Atoms, zlift: float = 2.0) -> Atoms:
    """Translate and rotate the adsorbate to the site for monodentate adsorption."""
    adsorbate = adsorbate.copy()

    anchor_position = adsorbate.info["anchor_position"]
    anchor_direction = adsorbate.info["anchor_direction"]
    assert np.allclose(anchor_direction, np.array([1.0, 0.0, 0.0])), "The anchor direction should point along +x."

    site_position = site["position"]

    # For monodentate, no need to compute rotation for the surface (xy) plane.

    # move the adsorbate to the site
    adsorbate.positions += site_position - anchor_position

    # lift the adsorbate a bit
    eps = 0.1  # to avoid zero division in up_direction calculation
    contact_index = adsorbate.info.get("anchor_index", 0)
    contact_position = adsorbate.positions[contact_index] + np.array([0.0, 0.0, eps])
    up_direction = contact_position - site_position  # from site to C atom
    up_direction = up_direction / np.linalg.norm(up_direction)

    # Align adsorbate direction to site normal if adsorbate is a molecule
    num_atoms_in_adsorbate = len(adsorbate)
    if num_atoms_in_adsorbate > 1:
        site_normal = site["normal"]
        angle = np.arccos(np.dot(up_direction, site_normal)) / np.pi * 180.0
        if np.dot(np.cross(up_direction, site_normal), anchor_direction) < 0:
            angle = 360 - angle  # according to the anchor direction
        adsorbate.rotate(angle, anchor_direction, center=contact_position)
        up_direction = site_normal  # update up_direction

    # Lift the adsorbate
    adsorbate.positions += zlift * up_direction

    return adsorbate


def prepare_bidentate_adsorbate(site, adsorbate: Atoms, zlift: float = 2.0) -> Atoms:
    """Translate and rotate the adsorbate to the site for bidentate adsorption."""
    adsorbate = adsorbate.copy()

    anchor_position = adsorbate.info["anchor_position"]
    anchor_direction = adsorbate.info["anchor_direction"]
    assert np.allclose(anchor_direction, np.array([1.0, 0.0, 0.0])), "The anchor direction should point along +x."

    site_position = site["position"]
    site_direction = site["direction"]

    # normalise directions
    site_direction = site_direction / np.linalg.norm(site_direction)

    # decompose site_direction into xy plane and z direction
    site_direction_z = np.array([0.0, 0.0, site_direction[2]])
    site_direction_xy = site_direction - site_direction_z
    site_direction_z = site_direction_z / np.linalg.norm(site_direction_z)
    site_direction_xy = site_direction_xy / np.linalg.norm(site_direction_xy)

    # compute rotation for the adsorbate plane
    if np.fabs(site_direction[2]) > 0.10:
        # v1 = adsorbate.positions[2] - adsorbate.positions[0]
        # v2 = adsorbate.positions[3] - adsorbate.positions[0]
        # plane_normal = np.cross(v1, v2)  # right hand rule
        # plane_normal = plane_normal / np.linalg.norm(plane_normal)
        plane_normal = adsorbate.info["molecular_plane_normal"]
        assert np.allclose(
            plane_normal, np.array([0.0, 1.0, 0.0])
        ), "The molecualr plane normal should point along +y."

        angle = 90 - np.arccos(np.dot(site_direction_z, site_direction)) / np.pi * 180.0
        if site_direction_z[2] > 0:
            angle = 360 - angle  # according to the plane normal direction
        adsorbate.rotate(angle, plane_normal, center=anchor_position)
    else:
        ...

    # compute rotation for the surface (xy) plane
    angle = np.arccos(np.dot(site_direction_xy, anchor_direction)) / np.pi * 180.0
    if site_direction_xy[1] < 0:
        angle = 360 - angle  # according to the y direction
    adsorbate.rotate(angle, "z", center=anchor_position)

    # move the adsorbate to the site
    adsorbate.positions += site_position - anchor_position

    # lift the adsorbate a bit
    contact_index = adsorbate.info.get("contact_index", 0)
    up_direction = adsorbate.positions[contact_index] - site_position  # from site to C atom
    up_direction = up_direction / np.linalg.norm(up_direction)
    adsorbate.positions += zlift * up_direction

    return adsorbate


def remove_one_particle(
    atoms: Atoms,
    identities: dict,
    species: str,
    sort_tags: bool = True,
    rng: np.random.Generator = np.random.default_rng(),
) -> tuple[Atoms, str]:
    """Remove one particle from the given atoms.

    Args:
        atoms: The input structure.
        identities: A dict with particles names and atomic indices.
        species: The particle name.
        sort_tags: Whether sort atoms by tags.
        rng: The random number generator.

    Returns:
        The new structure and the auxiliary information.

    """
    num_particles = len(identities[species])
    selected = rng.choice(range(num_particles), replace=False)
    selected_indices = identities[species][selected][1]
    del atoms[selected_indices]

    if sort_tags:
        atoms = reassign_tags_by_species(atoms)

    return atoms, f"remove_{species}_{selected_indices}"


def insert_one_particle(
    atoms: Atoms,
    particle: Atoms,
    region,
    covalent_ratio,
    bond_distance_dict,
    particle_tag: Optional[int] = None,
    sort_tags: bool = True,
    max_attempts: int = 100,
    check_distance_func: Optional[Callable] = check_atomic_distances,
    copy_atoms: bool = True,
    rng: np.random.Generator = np.random.default_rng(),
) -> tuple[Optional[Atoms], str]:
    """"""
    # Set the tag for the inserted particle,
    # which should not be used in atoms.
    if particle_tag is None:
        particle_tag = int(np.max(atoms.get_tags()) + 17)
    particle.set_tags(particle_tag)

    # Check if we should check neighbour distances
    num_atoms = len(atoms)

    if check_distance_func is not None:
        # Avoid distance check in the substrate and the particle to insert
        intra_bond_pairs = list(itertools.permutations(range(0, num_atoms), 2))
        intra_bond_pairs.extend(list(itertools.permutations(range(num_atoms, num_atoms + len(particle)), 2)))
        # We only check bond distances form by atoms in the particle,
        # since the existing atoms may not statisfy our distance criteria.
        atomic_indices = list(range(num_atoms, num_atoms + len(particle)))
        # Build the function
        post_func = functools.partial(
            check_distance_func,
            covalent_ratio=covalent_ratio,
            bond_distance_dict=bond_distance_dict,
            atomic_indices=atomic_indices,
            excluded_pairs=intra_bond_pairs,
            allow_isolated=False,
        )
    else:
        post_func = lambda _: True

    # Try inserting
    num_attempts = 0
    if copy_atoms:
        candidate = atoms + particle
    else:  # A revert is necessary if insert is used in MC.
        candidate = atoms
        candidate.extend(particle)
    region.preprocess(candidate)
    for iattempt in range(max_attempts):
        if copy_atoms and (len(atoms) != num_atoms):
            # Make sure we have not messed up with the substrate
            raise Exception(f"Expecting {num_atoms} but got {len(atoms)}.")
        position = region.get_random_positions(size=1, rng=rng)[0]
        new_particle = copy.deepcopy(particle)
        new_particle = translate_then_rotate(new_particle, position=position, use_com=True, rng=rng)
        candidate.positions[num_atoms:] = new_particle.positions
        if post_func(candidate):
            num_attempts = iattempt + 1
            break
    else:
        candidate = None
        num_attempts = max_attempts

    chemical_formula = particle.get_chemical_formula()
    state = "success"
    if candidate is not None:
        if sort_tags:
            candidate = reassign_tags_by_species(candidate)
    else:
        state = "failure"

    return candidate, f"insert_{chemical_formula}_{state}_{num_attempts}"


def insert_one_particle_on_site(
    atoms: Atoms,
    particle: Atoms,
    find_sites_func: Callable,
    covalent_ratio,
    bond_distance_dict,
    particle_tag: Optional[int] = None,
    sort_tags: bool = True,
    max_attempts: int = 100,
    check_distance_func: Optional[Callable] = check_atomic_distances,
    rng: np.random.Generator = np.random.default_rng(),
) -> tuple[Optional[Atoms], str]:
    """"""
    # Set the tag for the inserted particle,
    # which should not be used in atoms.
    if particle_tag is None:
        particle_tag = int(np.max(atoms.get_tags()) + 17)
    particle.set_tags(particle_tag)

    chemical_formula = particle.get_chemical_formula()

    anchor_mode = particle.info.get("anchor_mode")

    # Check if we should check neighbour distances
    num_atoms = len(atoms)

    if check_distance_func is not None:
        # Avoid distance check in the substrate and the particle to insert
        intra_bond_pairs = list(itertools.permutations(range(0, num_atoms), 2))
        intra_bond_pairs.extend(list(itertools.permutations(range(num_atoms, num_atoms + len(particle)), 2)))
        # We only check bond distances form by atoms in the particle,
        # since the existing atoms may not statisfy our distance criteria.
        atomic_indices_to_check = list(range(num_atoms, num_atoms + len(particle)))
        # Build the function
        post_func = functools.partial(
            check_distance_func,
            covalent_ratio=covalent_ratio,
            bond_distance_dict=bond_distance_dict,
            atomic_indices=atomic_indices_to_check,
            excluded_pairs=intra_bond_pairs,
            allow_isolated=False,
        )
    else:
        post_func = lambda _: True

    candidate = atoms

    # Get valid sites for various anchor modes
    sites = find_sites_func(candidate)
    if anchor_mode == "mono":
        anchor_func = prepare_monodentate_adsorbate
    elif anchor_mode == "bi":
        anchor_func = prepare_bidentate_adsorbate
        # TODO: support bridge sites only
        sites = [s for s in sites if s["type"] == "bridge"]
    else:
        raise Exception(f"Unknown anchor_mode `{anchor_mode}` should not happen.")

    num_sites = len(sites)
    if num_sites == 0:
        return None, f"ins_site_{chemical_formula}_nosites"

    used_sites = set()
    for iattempt in range(max_attempts):
        site_index = rng.integers(num_sites)
        site_identifier = tuple(sorted([n.idx for n in sites[site_index]["atoms"]]))
        if site_identifier in used_sites:
            continue
        # insert adsorbate
        new_particle = anchor_func(sites[site_index], particle)
        candidate.extend(new_particle)
        if post_func(candidate):  # geometric restraint
            num_attempts = iattempt + 1
            break
        else:
            del candidate[num_atoms:]  # revert
            used_sites.add(site_identifier)
    else:
        candidate = None
        num_attempts = max_attempts

    state = "success"
    if candidate is not None:
        if sort_tags:
            candidate = reassign_tags_by_species(candidate)
    else:
        state = "failure"

    return candidate, f"ins_site_{chemical_formula}_{state}_{num_attempts}"


if __name__ == "__main__":
    ...
