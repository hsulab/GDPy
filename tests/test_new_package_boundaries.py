def test_new_domain_packages_import_without_workflow_implementations():
    import sys

    import gdpx.analysis
    import gdpx.data
    import gdpx.exploration
    import gdpx.modifiers
    import gdpx.structures

    loaded = [name for name in sys.modules if name == "gdpx.nodes" or name.startswith("gdpx.nodes.")]
    assert not loaded


def test_scheduler_compatibility_export_is_lazy_and_resolves():
    from gdpx.execution.schedulers import LocalScheduler
    from gdpx.scheduler.local import LocalScheduler as LegacyLocalScheduler

    assert LocalScheduler is LegacyLocalScheduler

