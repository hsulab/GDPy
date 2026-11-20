"""Population atom roles, independent of fragment tags and calculation providers."""
import copy
from collections.abc import Mapping

import numpy as np

from gdpx.structures.groups import evaluate_group_expression, evaluate_constraint_expression
from gdpx.providers.configuration import resolve_executor_parameters

ROLES = ('active', 'buffer', 'frozen')
KEY = 'search_space'


def membership(atoms):
    """Read persisted membership; ASE databases store custom arrays in data."""
    saved = atoms.info.get('data', {}).get(KEY)
    result = {}
    for role in ROLES:
        array = atoms.arrays.get('gdpx_' + role)
        if array is None and saved is not None:
            array = saved[role]
        if array is None:
            return None
        if len(array) != len(atoms):
            raise ValueError('Search-space membership does not match atom count.')
        result[role] = np.asarray(array, dtype=bool)
    return result


def set_membership(atoms, groups):
    for role in ROLES:
        atoms.set_array('gdpx_' + role, np.asarray(groups[role], dtype=bool))
    persist_membership(atoms)


def persist_membership(atoms):
    groups = membership(atoms)
    if groups is not None:
        data = dict(atoms.info.get('data', {}))
        data[KEY] = {k: v.tolist() for k, v in groups.items()}
        atoms.info['data'] = data


def transfer_membership(source, target):
    groups = membership(source)
    if groups is not None:
        if len(source) != len(target) or not np.array_equal(source.numbers, target.numbers):
            raise ValueError('Calculation changed atom identity while transferring search-space membership.')
        set_membership(target, groups)


class SearchSpace:
    def __init__(self, definition=None):
        if definition is None:
            definition = {}
        if not isinstance(definition, Mapping) or set(definition) - set(ROLES):
            raise ValueError('system.search_space accepts active, buffer, and frozen group expressions.')
        self.definition = copy.deepcopy(dict(definition))
        for role, expression in self.definition.items():
            if expression is not None and (not isinstance(expression, str) or not expression.strip()):
                raise ValueError(f'system.search_space.{role} must be a group expression or null.')
        self.constraints = []

    def prepare_runtime(self, definition):
        """Supply a missing runtime constraint from an explicit frozen group."""
        result = copy.deepcopy(definition)
        frozen = self.definition.get('frozen')
        executor = result.get('executor', {})
        params = executor.get('parameters', {})
        if frozen is not None and params.get('setup', {}).get('constraint', params.get('constraint')) is None:
            params = executor.setdefault('parameters', {})
            if any(key in params for key in ('setup', 'stop', 'output')):
                params.setdefault('setup', {})['constraint'] = frozen
            else:
                params['constraint'] = frozen
        return result

    def bind_runtime(self, definition, simulation=None):
        executors = [definition['executor']]
        if simulation is not None:
            from ..simulated_annealing.protocol import simulation_executor
            executors.append(simulation_executor(simulation))
        self.constraints = [resolve_executor_parameters(e.get('parameters', {}), e['method']).get('constraint')
                            for e in executors]

    def resolve(self, atoms, *, reset=False):
        groups = None if reset else membership(atoms)
        if groups is None:
            groups = {}
            for role in ROLES:
                expr = self.definition.get(role)
                if role == 'active' and expr is None:
                    groups[role] = atoms.get_tags() != 0 if atoms.has('tags') else np.ones(len(atoms), bool)
                else:
                    mask = np.zeros(len(atoms), bool)
                    if expr is not None:
                        mask[evaluate_group_expression(atoms, expr)] = True
                    groups[role] = mask
            if self.definition.get('frozen') is None and self.constraints:
                groups['frozen'][evaluate_constraint_expression(atoms, self.constraints[0])[1]] = True
            if np.any(groups['active'] & groups['buffer']):
                raise ValueError('search_space.active and buffer must not overlap.')
            for role in ('active', 'buffer'):
                if self.definition.get(role) is not None and np.any(groups[role] & groups['frozen']):
                    raise ValueError(f'search_space.{role} and frozen must not overlap.')
            set_membership(atoms, groups)
        # Runtime expressions are compared by membership, never by string spelling.
        for constraint in self.constraints:
            if constraint is None and self.definition.get('frozen') is None and not groups['frozen'].any():
                continue
            indices = evaluate_constraint_expression(atoms, constraint)[1]
            if set(indices) != set(np.flatnonzero(groups['frozen'])):
                raise ValueError('search_space.frozen is inconsistent with runtime constraint; '
                                 'SA simulation and minimisation must freeze the same atoms.')
        set_membership(atoms, groups)
        return groups

    def select(self, atoms, expression=None, movable=False):
        groups = self.resolve(atoms)
        if expression is None:
            selected = groups['active'].copy()
        else:
            selected = np.zeros(len(atoms), bool)
            selected[evaluate_group_expression(atoms, expression)] = True
        if movable:
            selected &= ~groups['frozen']
        return np.flatnonzero(selected).tolist()
