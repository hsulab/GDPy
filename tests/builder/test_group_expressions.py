import pytest
from ase import Atoms
from gdpx.structures.groups import evaluate_group_expression, evaluate_constraint_expression


@pytest.mark.parametrize('expression,expected', [
    ('not tag 0', [1, 2, 3]), ('not (tag 0)', [1, 2, 3]), ('tag 0', [0]),
    ('tag 1:2', [1, 2]), ('tag -1', [3]),
    ('not not tag 0', [0]),
    ('not tag 0 and symbol Cu', [1, 3]),
    ('tag 0 or tag 2', [0, 2]),
    ('tag 0 or tag 1 and symbol O', [0]),
    ('(tag 0 or tag 1) and symbol O', []),
    ('not(tag 0 or tag 2)', [1, 3]),
    ('symbol Cu O', [0, 1, 2, 3]),
    ('index 0 2', [0, 2]), ('id 1:2', [0, 1]),
    ('region sphere 0 0 0 1', [0, 1, 2, 3]),
])
def test_group_expressions(expression, expected):
    atoms = Atoms('CuCuOCu', tags=[0, 1, 2, -1])
    assert sorted(evaluate_group_expression(atoms, expression)) == expected


@pytest.mark.parametrize('expression', [
    'tags != 0', 'tags == 0', 'tags > 1', '`tag 0`',
    'positions > 0', 'tag', 'symbol', 'tag 0 and', 'tag 0 or or tag 1',
    'not ()', '(tag 0', 'tag 0)', '', '   ',
    '__import__("os")', 'tags.__class__', 'missing', 'tag(0)',
])
def test_rejects_unsupported_expressions(expression):
    with pytest.raises((ValueError, TypeError)):
        evaluate_group_expression(Atoms('Cu'), expression)


def test_missing_tags_follow_ase_zero_default():
    assert evaluate_group_expression(Atoms('Cu'), 'not tag 0') == []


def test_selector_constraint_and_rejection_of_old_delimiters():
    atoms = Atoms('Cu2', tags=[0, 1])
    assert evaluate_constraint_expression(atoms, 'tag 0') == ([1], [0])
    with pytest.raises(ValueError, match='Remove backticks'):
        evaluate_constraint_expression(atoms, '`tag 0`')
