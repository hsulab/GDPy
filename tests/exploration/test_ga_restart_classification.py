from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.genetic_algorithm.generation import GeneticGenerationManager
from gdpx.exploration.persist.database import GlobalOptimisationDatabase


def manager():
    result = GeneticGenerationManager.__new__(GeneticGenerationManager)
    result._print = lambda *args: None
    return result


@pytest.mark.parametrize('generation,flags,data,group', [
    (1, {'pairing': 1}, {}, 'paired'),
    (1, {'mutation': 1}, {}, 'mutated'),
    (1, {}, {'builder': 'random'}, 'completion'),
    (0, {}, {'builder': 'random'}, 'initial'),
])
@pytest.mark.parametrize('origin', ['OperatorWithNoTypeInItsName', None])
def test_classification_uses_metadata(tmp_path, generation, flags, data, group, origin):
    db = GlobalOptimisationDatabase(tmp_path / 'candidates.db')
    a = Atoms('Cu', positions=[[0, 0, 0]])
    a.info['data'] = data
    if origin is not None:
        a.info['key_value_pairs'] = {'origin': origin}
    db.add_unrelaxed_candidate(a, generation=generation, **flags)
    groups = manager()._get_current_candidates(db, generation)
    assert {name: len(values) for name, values in groups.items()} == {
        name: int(name == group) for name in groups
    }


def test_mutated_reproduction_retains_group_and_latest_geometry(tmp_path):
    db = GlobalOptimisationDatabase(tmp_path / 'candidates.db')
    a = Atoms('Cu', positions=[[0, 0, 0]])
    a.info['key_value_pairs'] = {'origin': 'PeriodicCutAndSpliceCrossover'}
    db.add_unrelaxed_candidate(a, generation=1, pairing=1)
    b = a.copy()
    b.info = {'confid': a.info['confid'], 'key_value_pairs': {'origin': 'RenamedOperator'}}
    b.positions[:] = [1, 2, 3]
    db.add_unrelaxed_step(b, mutation=1)
    db.mark_as_queued(b)
    groups = manager()._get_current_candidates(db, 1)
    assert len(groups['paired']) == 1
    assert not groups['mutated']
    np.testing.assert_array_equal(groups['paired'][0].positions, b.positions)
    assert groups['paired'][0].info['confid'] == a.info['confid']


def test_unknown_production_metadata_fails_explicitly(tmp_path):
    db = GlobalOptimisationDatabase(tmp_path / 'candidates.db')
    a = Atoms('Cu')
    a.info['key_value_pairs'] = {'origin': 'LooksLikePairing'}
    db.add_unrelaxed_candidate(a, generation=1)
    with pytest.raises(RuntimeError, match='missing pairing, mutation, or builder metadata'):
        manager()._get_current_candidates(db, 1)
