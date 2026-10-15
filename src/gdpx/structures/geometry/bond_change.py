"""Single-atom bond changes with element pairs and distance bounds."""

from numbers import Real

import numpy as np
from ase.geometry import find_mic

from gdpx.structures.groups import evaluate_group_expression

from .fragments import FragmentPolicy, element_pair
from .spatial import check_atomic_distances, get_bond_distance_dict


BOND_CHANGE_VERSION = 4
OBSOLETE_BOND_ARGUMENTS = frozenset({
    "preset", "pairs", "mode", "min_cutoff", "max_cutoff", "form_distance",
    "break_distance", "max_pair_distance", "break_angle", "bond_ratio",
    "bond_cutoffs", "break_fragments", "form_fragments", "moving_particles", "departing_fragments",
})


def reject_obsolete_arguments(arguments):
    obsolete = OBSOLETE_BOND_ARGUMENTS.intersection(arguments)
    if obsolete:
        raise TypeError(f"Obsolete bond_change arguments: {', '.join(sorted(obsolete))}. "
                        "Use bond_break or bond_form with particles, bond, cutoffs and covalent_ratio; start a fresh BH run.")


def _bounds(values, name):
    if (not isinstance(values, (list, tuple)) or len(values) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in values)):
        raise ValueError(f"{name} must contain two finite numbers.")
    return [float(v) for v in values]


def _bond_settings(*, particles, bond, cutoffs=(0.0, 3.5), covalent_ratio=(0.8, 1.2)):
    """Validate and normalize the public bond-change arguments for serialization."""
    if (not isinstance(particles, (list, tuple)) or not particles
            or any(not isinstance(p, str) or not p for p in particles)):
        raise ValueError("particles must be a nonempty list of element symbols (single atoms).")
    for symbol in particles:
        element_pair((symbol, symbol))
    bond = list(element_pair(bond.split("-") if isinstance(bond, str) else bond))
    cutoffs = _bounds(cutoffs, "cutoffs")
    if not 0 <= cutoffs[0] < cutoffs[1]:
        raise ValueError("cutoffs must satisfy 0 <= min < max, in angstrom.")
    covalent_ratio = _bounds(covalent_ratio, "covalent_ratio")
    if not 0 < covalent_ratio[0] < 1 < covalent_ratio[1]:
        raise ValueError("covalent_ratio must satisfy 0 < lower < 1 < upper.")
    settings = dict(particles=sorted(set(particles)), bond=bond, cutoffs=cutoffs,
                    covalent_ratio=covalent_ratio)
    return settings


def bond_break_settings(*, particles, bond, cutoffs=(0.0, 3.5), covalent_ratio=(0.8, 1.2)):
    return _bond_settings(particles=particles, bond=bond, cutoffs=cutoffs, covalent_ratio=covalent_ratio)


def bond_form_settings(*, particles, bond, cutoffs=(0.0, 3.5),
                       covalent_ratio=(0.8, 1.2), acceptor_group=None):
    settings = _bond_settings(particles=particles, bond=bond, cutoffs=cutoffs,
                              covalent_ratio=covalent_ratio)
    if acceptor_group is not None:
        if not isinstance(acceptor_group, str) or not acceptor_group.strip():
            raise ValueError("acceptor_group must be a nonempty atom-group expression or None.")
        settings["acceptor_group"] = acceptor_group.strip()
    return settings


def retag_fragments(atoms, policy, preferred=None, before_move=None):
    """Reconcile geometry tags, notifying an optional undo recorder before edits."""
    tags = policy.tags_for(atoms, preferred)
    changed = np.flatnonzero(tags != atoms.get_tags())
    if len(changed):
        if before_move is not None:
            before_move(changed, names=("tags",))
        atoms.arrays["tags"][changed] = tags[changed]


def _find_bond_candidates(atoms, operation, *, particles, bond, cutoffs=(0.0, 3.5),
                          covalent_ratio=(0.8, 1.2), eligible_tags=None, acceptor_group=None):
    """Find terminal-atom breaks or isolated-atom joins inside the search window.

    Only selected positive-tag atoms in the region can move. A stationary
    partner may lie outside that region. Fixed atoms and substrate atoms never
    move. Tags must already reflect the current connectivity.
    """
    settings = _bond_settings(particles=particles, bond=bond, cutoffs=cutoffs,
                              covalent_ratio=covalent_ratio)
    if operation == "break":
        settings = bond_break_settings(**settings)
    else:
        settings = bond_form_settings(**settings, acceptor_group=acceptor_group)
    try:
        acceptors = set(evaluate_group_expression(atoms, acceptor_group)) if operation == "form" else set()
    except Exception as exc:
        raise ValueError(f"Invalid acceptor_group expression: {acceptor_group!r}") from exc
    if any(index < 0 or index >= len(atoms) for index in acceptors):
        raise ValueError("acceptor_group contains atom indices outside the structure.")
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
    selected = {i for i, (tag, symbol) in enumerate(zip(tags, symbols))
                if tag > 0 and tag in region_tags and symbol in settings["particles"]}
    found = []
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
            if i not in selected and j not in selected:
                continue
            distance = distances[i, j]
            if not minimum <= distance <= maximum or distance < lower:
                continue
            bonded = distance <= upper
            if bonded != (operation == "break"):
                continue
            # A degree-one atom becomes isolated when its only bond is cut;
            # a degree-zero atom can form a bond without carrying any neighbors.
            orientations = []
            for stationary, moving in ((i, j), (j, i)):
                if moving not in selected or moving in fixed:
                    continue
                if operation == "break":
                    movable = graph[moving] == {stationary}
                else:
                    movable = not graph[moving] and stationary in acceptors
                if movable:
                    orientations.append((stationary, moving, [moving]))
            if orientations:
                found.append(orientations)
    return found


def find_bond_break_candidates(atoms, *, eligible_tags=None, **settings):
    """Enumerate bridge bonds that can release one selected atom."""
    return _find_bond_candidates(atoms, "break", eligible_tags=eligible_tags,
                                 **bond_break_settings(**settings))


def find_bond_form_candidates(atoms, *, eligible_tags=None, **settings):
    """Enumerate joins from isolated selected atoms to stationary acceptors."""
    return _find_bond_candidates(atoms, "form", eligible_tags=eligible_tags,
                                 **bond_form_settings(**settings))


def break_bond(atoms, rng, *, particles=None, bond=None, cutoffs=(0.0, 3.5),
               covalent_ratio=(0.8, 1.2), **kwargs):
    """Break a bridge bond by moving one atom of a selected element.

    Only a terminal atom that becomes isolated after the cut can move. The
    rest of the structure stays stationary, regardless of fragment formulas.
    """
    reject_obsolete_arguments(kwargs)
    settings = bond_break_settings(particles=particles, bond=bond, cutoffs=cutoffs,
                                   covalent_ratio=covalent_ratio)
    return _change_bond(atoms, rng, "break", settings, **kwargs)


def form_bond(atoms, rng, *, particles=None, bond=None, cutoffs=(0.0, 3.5),
              covalent_ratio=(0.8, 1.2), acceptor_group=None, **kwargs):
    """Move one isolated selected atom onto an atom in ``acceptor_group``.

    Groups use the standard atom-group expression syntax. None permits all
    stationary endpoints; an empty selection permits no joins. Selected atoms
    already bonded to another atom cannot move during formation.
    """
    reject_obsolete_arguments(kwargs)
    settings = bond_form_settings(particles=particles, bond=bond, cutoffs=cutoffs,
                                  covalent_ratio=covalent_ratio, acceptor_group=acceptor_group)
    return _change_bond(atoms, rng, "form", settings, **kwargs)


def _change_bond(atoms, rng, operation, settings, *, eligible_tags=None, bond_distances=None,
                 allow_isolated=True, skip_distance_check=False,
                 max_random_attempts=1000, before_move=None):
    """Apply geometry only; restore coordinates and tags on failure or error."""
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
        candidates = _find_bond_candidates(atoms, operation, eligible_tags=eligible_tags, **settings)
        if not candidates:
            result = None, "BondChange_Skipped"
        else:
            distances = get_bond_distance_dict(sorted(set(atoms.numbers)), ratio=1.0)
            distances.update(bond_distances or {})
            threshold = policy.cutoff(settings["bond"])
            result = _move_bond_atom(atoms, rng, candidates, operation, policy,
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


def _move_bond_atom(atoms, rng, candidates, operation, policy, *, form_distance,
                        break_distance, covalent_ratio, bond_distances,
                        allow_isolated, skip_distance_check, max_random_attempts, before_move):
    choices = candidates[int(rng.choice(len(candidates)))]
    stationary, moving, indices = choices[int(rng.choice(len(choices)))]
    original = atoms.positions[moving].copy()
    old_tags = atoms.get_tags().copy()
    if before_move is not None:
        before_move(indices)
    direction, length = find_mic(atoms.positions[moving]-atoms.positions[stationary], atoms.cell, atoms.pbc)
    if not length > 0:
        return None, "BondChange_Failed"
    direction = direction / length
    distances = bond_distances
    for _ in range(max_random_attempts):
        if operation == "break":
            tangent = rng.normal(size=3)
            tangent -= np.dot(tangent, direction)*direction
            norm = np.linalg.norm(tangent)
            if not norm > 0:
                continue
            angle = np.deg2rad(rng.uniform(-10.0, 10.0))
            trial_direction = direction*np.cos(angle) + tangent/norm*np.sin(angle)
            target = atoms.positions[stationary] + break_distance*trial_direction
            atoms.positions[moving] = target
        else:
            trial_direction = rng.normal(size=3)
            norm = np.linalg.norm(trial_direction)
            if not norm > 0:
                continue
            atoms.positions[moving] = atoms.positions[stationary] + form_distance*trial_direction/norm
        bond_length = atoms.get_distance(stationary, moving, mic=True)
        cutoff = policy.cutoff((atoms[stationary].symbol, atoms[moving].symbol))
        correct_bond = bond_length > cutoff if operation == "break" else bond_length < cutoff
        target_distance = break_distance if operation == "break" else form_distance
        correct_bond = correct_bond and np.isclose(bond_length, target_distance, rtol=1e-10, atol=1e-10)
        if correct_bond and (skip_distance_check or check_atomic_distances(
            atoms, atomic_indices=indices, covalent_ratio=covalent_ratio,
            bond_distance_dict=distances, allow_isolated=allow_isolated,
        )):
            retag_fragments(atoms, policy, preferred=(stationary, int(old_tags[stationary])), before_move=before_move)
            changed = np.flatnonzero(old_tags != atoms.get_tags())
            event = dict(operation=operation, pair=[int(stationary), int(moving)], moved_indices=indices,
                         tag_indices=changed.tolist(), tags_before=old_tags[changed].tolist(),
                         tags_after=atoms.get_tags()[changed].tolist(), bond_distance=float(bond_length))
            return event, f"BondChange_{operation}_{stationary}_{moving}"
        atoms.positions[moving] = original
    return None, "BondChange_Failed"
