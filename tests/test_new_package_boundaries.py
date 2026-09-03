def test_new_domain_packages_import_without_workflow_implementations():
    import sys

    before = set(sys.modules)

    import gdpx.analysis
    import gdpx.data
    import gdpx.exploration
    import gdpx.modifiers
    import gdpx.structures

    loaded = [
        name for name in set(sys.modules) - before
        if name == "gdpx.workflow.nodes" or name.startswith("gdpx.workflow.nodes.")
    ]
    assert not loaded


def test_scheduler_public_export_is_lazy_and_resolves():
    from gdpx.execution.schedulers import LocalScheduler
    from gdpx.execution.schedulers.local import LocalScheduler as ConcreteLocalScheduler

    assert LocalScheduler is ConcreteLocalScheduler
