"""Adapt tag-based genetic operators to explicit population atom groups."""
import copy
import functools

import numpy as np

from .search_space import ROLES, set_membership


def bind_genetic_group(operator, policy, group=None):
    original = operator.get_new_individual

    @functools.wraps(original)
    def generate(parents):
        prepared = []
        maps = []
        for parent in parents:
            groups = policy.resolve(parent)
            selected = policy.select(parent, group, movable=True)
            if not selected:
                return None, 'mutation: empty search group'
            selected_set = set(selected)
            rest = [i for i in range(len(parent)) if i not in selected_set]
            tags = parent.get_tags()
            if getattr(operator, 'use_tags', False):
                for tag in set(tags[selected]) - {0}:
                    indices = set(np.flatnonzero(tags == tag))
                    if not indices <= selected_set:
                        raise ValueError('Operator group splits a tagged fragment; select the complete fragment.')
            order = rest + selected
            candidate = parent[order]
            candidate.info = copy.deepcopy(parent.info)
            temporary = tags[order].copy()
            temporary[:len(rest)] = 0
            tag_map = {}
            role_map = {}
            next_tag = max(0, int(tags.max())) + 1
            for offset, index in enumerate(selected, len(rest)):
                tag = int(tags[index])
                if tag == 0:
                    tag = next_tag
                    next_tag += 1
                    temporary[offset] = tag
                tag_map[tag] = int(tags[index])
                role_map[tag] = {role: bool(groups[role][index]) for role in ROLES}
            candidate.set_tags(temporary)
            prepared.append(candidate)
            maps.append((rest, order, tag_map, role_map, groups))
        rest, order, tag_map, role_map, groups = maps[0]
        # Crossovers use a separately held slab; supply the full excluded environment.
        old_slab = getattr(operator, 'slab', None)
        if hasattr(operator, 'slab'):
            operator.slab = prepared[0][:len(rest)]
        try:
            child, description = original(prepared)
        finally:
            if hasattr(operator, 'slab'):
                operator.slab = old_slab
        if child is None:
            return child, description
        tags = child.get_tags()
        excluded = np.flatnonzero(tags == 0)
        if len(excluded) != len(rest):
            raise ValueError('Genetic operator changed atoms outside its selected group.')
        child.positions[excluded] = parents[0].positions[rest]
        child.numbers[excluded] = parents[0].numbers[rest]
        output = {role: np.zeros(len(child), bool) for role in ROLES}
        for role in ROLES:
            output[role][excluded] = groups[role][rest]
        restored_tags = tags.copy()
        restored_tags[excluded] = parents[0].get_tags()[rest]
        reserved = set(parents[0].get_tags()[rest]) | set(tag_map.values())
        fresh_tags = {}
        for tag in set(tags) - {0} - set(tag_map):
            fresh = int(tag)
            if fresh in reserved:
                fresh = max(reserved | set(map(int, tags))) + 1
            fresh_tags[int(tag)] = fresh
            reserved.add(fresh)
        for i, tag in enumerate(tags):
            if tag == 0:
                continue
            restored_tags[i] = tag_map.get(int(tag), fresh_tags.get(int(tag), int(tag)))
            for role in ROLES:
                output[role][i] = role_map.get(int(tag), {'active': True}).get(role, False)
        child.set_tags(restored_tags)
        blmin = getattr(operator, 'blmin', None)
        if blmin and len(excluded) and len(child) > len(excluded):
            from gdpx.structures.geometry.ga import atoms_too_close_two_sets
            selected_child = [i for i, tag in enumerate(tags) if tag != 0]
            if atoms_too_close_two_sets(child[excluded], child[selected_child], blmin):
                return None, description
        set_membership(child, output)
        # Most operators preserve composition; restore original atom order as well.
        if len(child) == len(order) and np.array_equal(child.numbers, parents[0].numbers[order]):
            info = child.info
            child = child[np.argsort(order)]
            child.info = info
            set_membership(child, groups)
        return child, description

    operator.get_new_individual = generate
    return operator
