import pytest

from gdpx.modifiers import CommitteePotential, ModifiedPotential, PotentialLeaf, SumPotential, apply_modifiers
from gdpx.providers import ModifierSpec, PotentialSpec


def test_modifiers_preserve_declared_order():
    potential = PotentialSpec("emt")
    modifiers = (ModifierSpec("builtin", "first"), ModifierSpec("builtin", "second"))

    expression = apply_modifiers(potential, modifiers)

    assert isinstance(expression, ModifiedPotential)
    assert expression.modifier.method == "second"
    assert isinstance(expression.source, ModifiedPotential)
    assert expression.source.modifier.method == "first"
    assert isinstance(expression.source.source, PotentialLeaf)


def test_empty_compositions_are_rejected():
    with pytest.raises(ValueError):
        SumPotential(())
    with pytest.raises(ValueError):
        CommitteePotential(())

