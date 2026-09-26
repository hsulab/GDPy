def test_canonical_structure_namespaces():
    from gdpx.structures.builders.builder import StructureBuilder as ConcreteBuilder
    from gdpx.structures.builders import StructureBuilder
    from gdpx.structures.groups import evaluate_constraint_expression

    assert StructureBuilder is ConcreteBuilder
    assert callable(evaluate_constraint_expression)


def test_canonical_analysis_namespaces():
    from gdpx.analysis.comparators import BaseComparator
    from gdpx.analysis.descriptors import BaseDescriber
    from gdpx.analysis.selectors import BaseSelector
    from gdpx.analysis.validators import BaseValidator

    assert all(item is not None for item in (BaseComparator, BaseDescriber, BaseSelector, BaseValidator))


def test_canonical_execution_namespaces():
    from gdpx.execution.lifecycle import ComputePlan
    from gdpx.execution.workers import DriverBasedWorker

    assert ComputePlan is not None
    assert DriverBasedWorker is not None
