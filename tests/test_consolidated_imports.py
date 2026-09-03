def test_structure_compatibility_namespaces():
    from gdpx.builder.builder import StructureBuilder as LegacyBuilder
    from gdpx.structures.builders import StructureBuilder
    from gdpx.structures.groups import evaluate_constraint_expression

    assert StructureBuilder is LegacyBuilder
    assert callable(evaluate_constraint_expression)


def test_analysis_compatibility_namespaces():
    from gdpx.analysis.comparators import BaseComparator
    from gdpx.analysis.descriptors import BaseDescriber
    from gdpx.analysis.selectors import BaseSelector
    from gdpx.analysis.validators import BaseValidator

    assert all(item is not None for item in (BaseComparator, BaseDescriber, BaseSelector, BaseValidator))


def test_execution_compatibility_namespaces():
    from gdpx.execution.lifecycle import ComputePlan
    from gdpx.execution.workers import DriverBasedWorker, Pairing

    assert ComputePlan is not None
    assert DriverBasedWorker is not None
    assert Pairing.AUTO.name == "AUTO"

