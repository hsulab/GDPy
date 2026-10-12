"""Distance-based connectivity and stable movable-fragment identities."""

from collections import Counter
from numbers import Real

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list


def positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return float(value)


def element_pair(pair):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(
        not isinstance(symbol, str) or symbol not in atomic_numbers or symbol == "X" for symbol in pair
    ):
        raise ValueError("Bond pairs must contain two element symbols.")
    return tuple(sorted(pair))


def component(graph, start, removed=None):
    """Return a connected component, optionally omitting one undirected edge."""
    seen, pending = {start}, [start]
    while pending:
        current = pending.pop()
        for other in graph[current]:
            if removed is not None and frozenset((current, other)) == removed:
                continue
            if other not in seen:
                seen.add(other)
                pending.append(other)
    return seen


class FragmentPolicy:
    """Infer chemical fragments without absorbing mobile atoms into tag 0."""

    def __init__(self, bond_ratio=1.2, bond_cutoffs=None):
        self.bond_ratio = positive_number(bond_ratio, "bond_ratio")
        if bond_cutoffs is not None and not isinstance(bond_cutoffs, dict):
            raise ValueError("bond_cutoffs must map element-pair strings to distances.")
        self.bond_cutoffs = {}
        for key, distance in (bond_cutoffs or {}).items():
            if not isinstance(key, str):
                raise ValueError("Use element-pair keys such as 'O-H' in bond_cutoffs.")
            pair = element_pair(key.split("-"))
            if pair in self.bond_cutoffs:
                raise ValueError("Duplicate bond cutoff for " + key)
            self.bond_cutoffs[pair] = positive_number(distance, "bond_cutoffs")

    def as_dict(self):
        return dict(bond_ratio=self.bond_ratio,
                    bond_cutoffs={"-".join(pair): value for pair, value in sorted(self.bond_cutoffs.items())})

    def cutoff(self, pair):
        pair = element_pair(pair)
        return self.bond_cutoffs.get(pair, self.bond_ratio * sum(covalent_radii[atomic_numbers[s]] for s in pair))

    def graph(self, atoms, include_substrate=False):
        if not atoms.has("tags") or np.any(atoms.get_tags() < 0):
            raise ValueError("bond_change requires nonnegative tags: substrate 0, mobile fragments positive.")
        symbols, tags = atoms.get_chemical_symbols(), atoms.get_tags()
        types = sorted(set(symbols))
        cutoffs = {(a, b): self.cutoff((a, b)) for a in types for b in types}
        first, second = neighbor_list("ij", atoms, cutoffs, self_interaction=False)
        graph = [set() for _ in atoms]
        edges = set()
        for i, j in zip(first.tolist(), second.tolist()):
            if i == j or (not include_substrate and (tags[i] == 0 or tags[j] == 0)):
                continue
            graph[i].add(j)
            graph[j].add(i)
            edges.add(tuple(sorted((i, j))))
        if include_substrate:
            # All substrate atoms represent one immovable body, even if the
            # distance graph contains a vacuum gap or disconnected pieces.
            substrate = np.flatnonzero(tags == 0).tolist()
            for i, j in zip(substrate, substrate[1:]):
                graph[i].add(j)
                graph[j].add(i)
        return graph, sorted(edges)

    def tags_for(self, atoms, preferred=None):
        graph, _ = self.graph(atoms)
        old = atoms.get_tags()
        unseen = set(np.flatnonzero(old > 0).tolist())
        groups = []
        while unseen:
            group = sorted(component(graph, min(unseen)))
            groups.append(group)
            unseen.difference_update(group)
        assigned, used = {}, set()
        if preferred is not None:
            index, tag = preferred
            if old[index] > 0:
                for number, group in enumerate(groups):
                    if index in group:
                        assigned[number] = int(tag)
                        used.add(int(tag))
                        break
        # Largest overlap wins; ties prefer the lower old tag, then atom index.
        overlaps = sorted((-count, int(tag), group[0], number)
                          for number, group in enumerate(groups)
                          for tag, count in Counter(old[group].tolist()).items())
        for _, tag, _, number in overlaps:
            if number not in assigned and tag not in used:
                assigned[number] = tag
                used.add(tag)
        fresh = int(old.max(initial=0)) + 1
        result = old.copy()
        for number, group in enumerate(groups):
            if number not in assigned:
                assigned[number] = fresh
                fresh += 1
            result[group] = assigned[number]
        return result

    def reconcile(self, atoms):
        old = atoms.get_tags()
        tags = self.tags_for(atoms)
        changed = np.flatnonzero(old != tags)
        atoms.set_tags(tags)
        _, edges = self.graph(atoms, include_substrate=True)
        record = dict(indices=changed.tolist(), before=old[changed].tolist(), after=tags[changed].tolist(),
                      bonds=[list(pair) for pair in edges if any(tags[i] > 0 for i in pair)])
        atoms.info["fragment_changes"] = record
        return record


def fragment_policy_for(operators):
    policies = [op.fragment_policy for op in operators if op.name == "bond_change"]
    if not policies:
        return None
    if any(op.name not in {"bond_change", "rotate", "move", "rattle", "bounce"} for op in operators):
        raise ValueError("bond_change searches currently require atom-order-preserving geometry operators.")
    if any(policy.as_dict() != policies[0].as_dict() for policy in policies[1:]):
        raise ValueError("All bond_change operators in a search must use the same fragment policy.")
    return policies[0]
