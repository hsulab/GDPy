"""Bond changes using fragment species, an element pair and distance bounds."""

from numbers import Real

import numpy as np
from ase.formula import Formula
from ase.geometry import find_mic

from .fragments import FragmentPolicy, component, element_pair
from .spatial import check_atomic_distances, get_bond_distance_dict


BOND_CHANGE_VERSION = 2
OBSOLETE_BOND_ARGUMENTS = frozenset({
    "preset", "pairs", "mode", "min_cutoff", "max_cutoff", "form_distance",
    "break_distance", "max_pair_distance", "break_angle", "bond_ratio",
    "bond_cutoffs", "break_fragments", "form_fragments",
})


def reject_obsolete_arguments(arguments):
    obsolete = OBSOLETE_BOND_ARGUMENTS.intersection(arguments)
    if obsolete:
        raise TypeError(f"Obsolete bond_change arguments: {', '.join(sorted(obsolete))}. "
                        "Use particles, bond, cutoffs and covalent_ratio; start a fresh BH run.")


def _bounds(values, name):
    if (not isinstance(values, (list, tuple)) or len(values) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in values)):
        raise ValueError(f"{name} must contain two finite numbers.")
    return [float(v) for v in values]


def bond_change_settings(*, particles, bond, cutoffs=(0.0, 3.5), covalent_ratio=(0.8, 1.2)):
    """Validate and normalize the public bond-change arguments for serialization."""
    if (not isinstance(particles, (list, tuple)) or not particles
            or any(not isinstance(p, str) or not p for p in particles)):
        raise ValueError("particles must be a nonempty list of molecular formulas.")
    species = []
    for value in particles:
        formula = Formula(value)
        if not formula.count() or any(count <= 0 for count in formula.count().values()):
            raise ValueError("particles must contain nonempty molecular formulas.")
        for symbol in formula.count():
            element_pair((symbol, symbol))
        species.append(formula.format("hill"))
    bond = list(element_pair(bond.split("-") if isinstance(bond, str) else bond))
    cutoffs = _bounds(cutoffs, "cutoffs")
    if not 0 <= cutoffs[0] < cutoffs[1]:
        raise ValueError("cutoffs must satisfy 0 <= min < max, in angstrom.")
    covalent_ratio = _bounds(covalent_ratio, "covalent_ratio")
    if not 0 < covalent_ratio[0] < 1 < covalent_ratio[1]:
        raise ValueError("covalent_ratio must satisfy 0 < lower < 1 < upper.")
    return dict(particles=sorted(set(species)), bond=bond, cutoffs=cutoffs,
                covalent_ratio=covalent_ratio)


def retag_fragments(atoms, policy, preferred=None, before_move=None):
    """Reconcile geometry tags, notifying an optional undo recorder before edits."""
    tags = policy.tags_for(atoms, preferred)
    changed = np.flatnonzero(tags != atoms.get_tags())
    if len(changed):
        if before_move is not None:
            before_move(changed, names=("tags",))
        atoms.arrays["tags"][changed] = tags[changed]


def find_bond_change_candidates(atoms, *, particles, bond, cutoffs=(0.0, 3.5),
                                covalent_ratio=(0.8, 1.2), eligible_tags=None):
    """Find bridge breaks and intercomponent joins inside an absolute window.

    Tags must describe current fragments. At least one endpoint must belong to
    a selected positive-tag species in ``eligible_tags`` (all tags by default).
    Its partner may lie outside the region or in the substrate. Ratios classify
    the selected pair; fixed components, cycles and intracomponent joins remain
    excluded. Both cutoff boundaries and the bonded upper bound are inclusive.
    """
    settings = bond_change_settings(particles=particles, bond=bond, cutoffs=cutoffs,
                                    covalent_ratio=covalent_ratio)
    bond = tuple(settings["bond"])
    policy = FragmentPolicy(bond_ratio=settings["covalent_ratio"][1])
    graph, _ = policy.graph(atoms, include_substrate=True)
    tags, symbols = atoms.get_tags(), atoms.get_chemical_symbols()
    fixed = set(np.flatnonzero(tags == 0).tolist())
    for constraint in atoms.constraints:
        if not hasattr(constraint, "get_indices"):
            raise ValueError("bond_change requires constraints exposing fixed atom indices.")
        fixed.update(constraint.get_indices().tolist())
    region_tags = set(np.unique(tags)) if eligible_tags is None else set(eligible_tags)
    selected = {int(tag) for tag in np.unique(tags) if tag > 0 and tag in region_tags
                and atoms[np.flatnonzero(tags == tag)].get_chemical_formula() in settings["particles"]}
    groups = {}
    for i in range(len(atoms)):
        if i not in groups:
            group = component(graph, i)
            for index in group:
                groups[index] = group
    found = {"break": [], "form": []}
    if not selected:
        return found
    distances = atoms.get_all_distances(mic=True)
    radius_sum = policy.cutoff(bond) / settings["covalent_ratio"][1]
    lower, upper = np.asarray(settings["covalent_ratio"]) * radius_sum
    minimum, maximum = settings["cutoffs"]
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            if tuple(sorted((symbols[i], symbols[j]))) != bond:
                continue
            if tags[i] not in selected and tags[j] not in selected:
                continue
            distance = distances[i, j]
            if not minimum <= distance <= maximum or distance < lower:
                continue
            if distance <= upper:
                left = component(graph, i, frozenset((i, j)))
                if j in left:
                    continue
                right = component(graph, j, frozenset((i, j)))
                operation = "break"
            else:
                if j in groups[i]:
                    continue
                left, right, operation = groups[i], groups[j], "form"
            orientations = []
            if not right.intersection(fixed):
                orientations.append((i, j, sorted(right)))
            if not left.intersection(fixed):
                orientations.append((j, i, sorted(left)))
            if orientations:
                found[operation].append(orientations)
    return found


def change_bond(atoms, rng, *, particles=None, bond=None, cutoffs=(0.0, 3.5),
                covalent_ratio=(0.8, 1.2), eligible_tags=None, bond_distances=None,
                allow_isolated=True, skip_distance_check=False,
                max_random_attempts=1000, before_move=None, **obsolete):
    """Change one bond in place; return ``(event, diagnostic)``.

    Search within ``cutoffs`` in angstrom. Relative to the covalent-radius sum,
    the lower ratio rejects overlaps and the upper ratio defines connectivity.
    Form at the radius sum; break at 1.25 times the upper bonding threshold.
    Failure or exceptions restore coordinates and tags. Successful edits remain
    in place, without modifying calculator, constraints, metadata or atom order.
    ``before_move`` can record touched rows for a caller-owned transaction.
    """
    reject_obsolete_arguments(obsolete)
    if obsolete:
        raise TypeError(f"Unknown bond_change arguments: {', '.join(sorted(obsolete))}")
    settings = bond_change_settings(particles=particles, bond=bond, cutoffs=cutoffs,
                                    covalent_ratio=covalent_ratio)
    if (isinstance(max_random_attempts, bool) or not isinstance(max_random_attempts, (int, np.integer))
            or max_random_attempts < 1):
        raise ValueError("max_random_attempts must be a positive integer.")
    covalent_ratio = settings["covalent_ratio"]
    policy = FragmentPolicy(bond_ratio=covalent_ratio[1])
    undo = []

    def record(indices, names=("positions",)):
        indices = np.asarray(indices, dtype=int)
        undo.append((indices, {name: atoms.arrays[name][indices].copy() for name in names}))
        if before_move is not None:
            before_move(indices, names=names)

    def restore():
        for indices, arrays in reversed(undo):
            for name, values in arrays.items():
                atoms.arrays[name][indices] = values

    try:
        retag_fragments(atoms, policy, before_move=record)
        candidates = find_bond_change_candidates(atoms, eligible_tags=eligible_tags, **settings)
        modes = [mode for mode, choices in candidates.items() if choices]
        if not modes:
            result = None, "BondChange_Skipped"
        else:
            distances = get_bond_distance_dict(sorted(set(atoms.numbers)), ratio=1.0)
            distances.update(bond_distances or {})
            threshold = policy.cutoff(settings["bond"])
            result = _move_bond_fragment(atoms, rng, candidates, modes, policy,
                form_distance=threshold/covalent_ratio[1], break_distance=1.25*threshold,
                covalent_ratio=covalent_ratio, bond_distances=distances,
                allow_isolated=allow_isolated, skip_distance_check=skip_distance_check,
                max_random_attempts=max_random_attempts, before_move=record)
        if result[0] is not None:
            return result
    except BaseException:
        restore()
        raise
    restore()
    return result


def _move_bond_fragment(atoms, rng, candidates, modes, policy, *, form_distance,
                        break_distance, covalent_ratio, bond_distances,
                        allow_isolated, skip_distance_check, max_random_attempts, before_move):
    mode = modes[int(rng.choice(len(modes)))]
    choices = candidates[mode][int(rng.choice(len(candidates[mode])))]
    stationary, moving, indices = choices[int(rng.choice(len(choices)))]
    original = atoms.positions[indices].copy()
    old_tags = atoms.get_tags().copy()
    if before_move is not None:
        before_move(indices)
    direction, length = find_mic(atoms.positions[moving]-atoms.positions[stationary], atoms.cell, atoms.pbc)
    if not length > 0:
        return None, "BondChange_Failed"
    direction = direction / length
    relative, _ = find_mic(original-atoms.positions[moving], atoms.cell, atoms.pbc)
    distances = bond_distances
    for _ in range(max_random_attempts):
        if mode == "break":
            tangent = rng.normal(size=3)
            tangent -= np.dot(tangent, direction)*direction
            norm = np.linalg.norm(tangent)
            if not norm > 0:
                continue
            angle = np.deg2rad(rng.uniform(-10.0, 10.0))
            trial_direction = direction*np.cos(angle) + tangent/norm*np.sin(angle)
            target = atoms.positions[stationary] + break_distance*trial_direction
            atoms.positions[indices] = original + target-atoms.positions[moving]
        else:
            axis = rng.normal(size=3)
            norm = np.linalg.norm(axis)
            if not norm > 0:
                continue
            axis /= norm
            angle = rng.uniform(-np.pi, np.pi)
            rotated = (relative*np.cos(angle) + np.cross(axis, relative)*np.sin(angle)
                       + np.outer(relative@axis, axis)*(1-np.cos(angle)))
            trial_direction = rng.normal(size=3)
            norm = np.linalg.norm(trial_direction)
            if not norm > 0:
                continue
            target = atoms.positions[stationary] + form_distance*trial_direction/norm
            atoms.positions[indices] = original + rotated-relative + target-atoms.positions[moving]
        bond_length = atoms.get_distance(stationary, moving, mic=True)
        cutoff = policy.cutoff((atoms[stationary].symbol, atoms[moving].symbol))
        correct_bond = bond_length > cutoff if mode == "break" else bond_length < cutoff
        target_distance = break_distance if mode == "break" else form_distance
        correct_bond = correct_bond and np.isclose(bond_length, target_distance, rtol=1e-10, atol=1e-10)
        if correct_bond and (skip_distance_check or check_atomic_distances(
            atoms, atomic_indices=indices, covalent_ratio=covalent_ratio,
            bond_distance_dict=distances, allow_isolated=allow_isolated,
            excluded_pairs=[(i, j) for i in indices for j in indices if i != j],
        )):
            retag_fragments(atoms, policy, preferred=(stationary, int(old_tags[stationary])), before_move=before_move)
            changed = np.flatnonzero(old_tags != atoms.get_tags())
            event = dict(operation=mode, pair=[int(stationary), int(moving)], moved_indices=indices,
                         tag_indices=changed.tolist(), tags_before=old_tags[changed].tolist(),
                         tags_after=atoms.get_tags()[changed].tolist(), bond_distance=float(bond_length))
            return event, f"BondChange_{mode}_{stationary}_{moving}"
        atoms.positions[indices] = original
    return None, "BondChange_Failed"
