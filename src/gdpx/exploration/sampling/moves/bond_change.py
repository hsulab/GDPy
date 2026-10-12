"""Geometry proposals for breaking bridges and joining separate fragments."""

import numpy as np
from ase.formula import Formula
from ase.geometry import find_mic

from gdpx.structures.geometry.spatial import check_atomic_distances

from ..fragments import FragmentPolicy, component, element_pair, positive_number
from .operator import BaseMCOperator


class BondChangeOperator(BaseMCOperator):
    """Break/form specified element pairs, preserving fragment internal geometry.

    These biased geometry moves target basin-hopping searches, not equilibrium
    reaction sampling. Bonds are inferred from distances, not sent to a force field.
    """

    name = "bond_change"

    def __init__(self, pairs, mode="both", form_distance=1.0, break_distance=2.5,
                 max_pair_distance=3.5, break_angle=10.0, bond_ratio=1.2, bond_cutoffs=None,
                 break_fragments=None, form_fragments=None, **kwargs):
        if not isinstance(pairs, (list, tuple)) or not pairs:
            raise ValueError("pairs must be a nonempty list of element pairs.")
        self.pairs = sorted(set(element_pair(pair) for pair in pairs))
        if mode not in ("break", "form", "both"):
            raise ValueError("mode must be break, form, or both.")
        self.mode = mode
        self.form_distance = positive_number(form_distance, "form_distance")
        self.break_distance = positive_number(break_distance, "break_distance")
        self.max_pair_distance = positive_number(max_pair_distance, "max_pair_distance")
        self.break_angle = positive_number(break_angle, "break_angle")
        if self.break_angle > 180:
            raise ValueError("break_angle must be at most 180 degrees.")
        self.fragment_policy = FragmentPolicy(bond_ratio, bond_cutoffs)
        for pair in self.pairs:
            cutoff = self.fragment_policy.cutoff(pair)
            if (mode != "break" and self.form_distance >= cutoff) or (mode != "form" and self.break_distance <= cutoff):
                raise ValueError("Enabled target distances must satisfy form_distance < bond cutoff < break_distance.")
        if mode != "break" and self.max_pair_distance < self.form_distance:
            raise ValueError("max_pair_distance must be at least form_distance.")
        self.break_fragments = self._formulas(break_fragments)
        if form_fragments is not None and (
            not isinstance(form_fragments, (list, tuple)) or not form_fragments
            or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in form_fragments)
        ):
            raise ValueError("form_fragments must list pairs of fragment formulas (or 'substrate').")
        self.form_fragments = None if form_fragments is None else [sorted(self._formulas(pair)) for pair in form_fragments]
        super().__init__(**kwargs)

    @staticmethod
    def _formulas(values):
        if values is None:
            return None
        if not isinstance(values, (list, tuple)) or not values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError("Fragment filters must be nonempty lists of formulas.")
        result = []
        for value in values:
            if value == "substrate":
                result.append(value)
            else:
                formula = Formula(value)
                # Validate element names as well as the formula syntax.
                for symbol in formula.count():
                    element_pair((symbol, symbol))
                result.append(formula.format("hill"))
        return result

    def _retag(self, atoms, preferred=None):
        tags = self.fragment_policy.tags_for(atoms, preferred)
        changed = np.flatnonzero(tags != atoms.get_tags())
        if len(changed):
            self._transaction.watch(changed, names=("tags",))
            atoms.arrays["tags"][changed] = tags[changed]

    def _candidates(self, atoms):
        graph, edges = self.fragment_policy.graph(atoms, include_substrate=True)
        tags, symbols = atoms.get_tags(), atoms.get_chemical_symbols()
        fixed = set(np.flatnonzero(tags == 0).tolist())
        for constraint in atoms.constraints:
            if not hasattr(constraint, "get_indices"):
                raise ValueError("bond_change requires constraints exposing fixed atom indices.")
            fixed.update(constraint.get_indices().tolist())
        eligible = {tag for values in self._curr_tags_dict.values() for tag in values}
        formulas = {int(tag): atoms[np.flatnonzero(tags == tag)].get_chemical_formula()
                    for tag in np.unique(tags) if tag > 0}
        formulas[0] = "substrate"
        groups = {}
        for i in range(len(atoms)):
            if i not in groups:
                group = component(graph, i)
                for index in group:
                    groups[index] = group
        found = {"break": [], "form": []}
        distances = atoms.get_all_distances(mic=True)
        edges = set(edges)
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                if tuple(sorted((symbols[i], symbols[j]))) not in self.pairs:
                    continue
                if tags[i] not in eligible or tags[j] not in eligible:
                    continue
                if (i, j) in edges:
                    if self.mode == "form" or (self.break_fragments is not None and not any(
                        formulas[int(tags[k])] in self.break_fragments for k in (i, j)
                    )):
                        continue
                    left = component(graph, i, frozenset((i, j)))
                    if j in left:  # Ring or alternative path through the substrate.
                        continue
                    right = component(graph, j, frozenset((i, j)))
                    operation = "break"
                else:
                    if self.mode == "break" or distances[i, j] > self.max_pair_distance or j in groups[i]:
                        continue
                    if self.form_fragments is not None and sorted((formulas[int(tags[i])], formulas[int(tags[j])])) not in self.form_fragments:
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

    def _propose(self, atoms, rng):
        self._retag(atoms)
        super()._propose(atoms, rng)
        candidates = self._candidates(atoms)
        modes = [mode for mode, choices in candidates.items() if choices]
        if not modes:
            self._extra_info = "BondChange_Skipped"
            return None
        mode = modes[int(rng.choice(len(modes)))]
        choices = candidates[mode][int(rng.choice(len(candidates[mode])))]
        stationary, moving, indices = choices[int(rng.choice(len(choices)))]
        original = atoms.positions[indices].copy()
        old_tags = atoms.get_tags().copy()
        self._transaction.watch(indices)
        direction, length = find_mic(atoms.positions[moving]-atoms.positions[stationary], atoms.cell, atoms.pbc)
        if not length > 0:
            self._extra_info = "BondChange_Failed"
            return None
        direction = direction / length
        relative, _ = find_mic(original-atoms.positions[moving], atoms.cell, atoms.pbc)
        distances = dict(getattr(self, "bond_distance_dict", {}))
        distances.update(getattr(self, "custom_pair_distance_dict", None) or {})
        for _ in range(self.MAX_RANDOM_ATTEMPTS):
            if mode == "break":
                tangent = rng.normal(size=3)
                tangent -= np.dot(tangent, direction)*direction
                norm = np.linalg.norm(tangent)
                if not norm > 0:
                    continue
                angle = np.deg2rad(rng.uniform(-self.break_angle, self.break_angle))
                trial_direction = direction*np.cos(angle) + tangent/norm*np.sin(angle)
                target = atoms.positions[stationary] + self.break_distance*trial_direction
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
                target = atoms.positions[stationary] + self.form_distance*trial_direction/norm
                atoms.positions[indices] = original + rotated-relative + target-atoms.positions[moving]
            bond_length = atoms.get_distance(stationary, moving, mic=True)
            cutoff = self.fragment_policy.cutoff((atoms[stationary].symbol, atoms[moving].symbol))
            correct_bond = bond_length > cutoff if mode == "break" else bond_length < cutoff
            target_distance = self.break_distance if mode == "break" else self.form_distance
            correct_bond = correct_bond and np.isclose(bond_length, target_distance, rtol=1e-10, atol=1e-10)
            if correct_bond and (self.skip_distance_check or check_atomic_distances(
                atoms, atomic_indices=indices, covalent_ratio=(self.covalent_min, self.covalent_max),
                bond_distance_dict=distances, allow_isolated=self.allow_isolated,
                excluded_pairs=[(i, j) for i in indices for j in indices if i != j],
            )):
                self._retag(atoms, preferred=(stationary, int(old_tags[stationary])))
                changed = np.flatnonzero(old_tags != atoms.get_tags())
                event = dict(operation=mode, pair=[int(stationary), int(moving)], moved_indices=indices,
                             tag_indices=changed.tolist(), tags_before=old_tags[changed].tolist(),
                             tags_after=atoms.get_tags()[changed].tolist(), bond_distance=float(bond_length))
                self._transaction.set_info("bond_change", event)
                self._state["bond_change"] = event
                self._extra_info = f"BondChange_{mode}_{stationary}_{moving}"
                return atoms
            atoms.positions[indices] = original
        self._extra_info = "BondChange_Failed"
        return None

    def as_dict(self):
        return dict(super().as_dict(), pairs=[list(pair) for pair in self.pairs], mode=self.mode,
                    form_distance=self.form_distance, break_distance=self.break_distance,
                    max_pair_distance=self.max_pair_distance, break_angle=self.break_angle,
                    break_fragments=self.break_fragments, form_fragments=self.form_fragments,
                    **self.fragment_policy.as_dict())
