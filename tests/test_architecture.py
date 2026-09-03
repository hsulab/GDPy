"""Regression tests for the domain/workflow dependency boundary."""

import ast
import copy
import os
import pathlib
import subprocess
import sys

import networkx as nx


ROOT = pathlib.Path(__file__).parents[1]
DOMAIN_DIRECTORIES = tuple(
    path.name
    for path in (ROOT / "src" / "gdpx").iterdir()
    if path.is_dir()
    and (path / "__init__.py").exists()
    and path.name not in {"cli", "nodes", "session", "workflow"}
)


def test_domain_modules_do_not_import_workflow_layers():
    violations = []
    for directory in DOMAIN_DIRECTORIES:
        for path in (ROOT / "src" / "gdpx" / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name == "gdpx.workflow.nodes" or name.startswith("gdpx.workflow.nodes.") or name == "gdpx.workflow.session" or name.startswith("gdpx.workflow.session."):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "Domain-to-workflow imports found:\n" + "\n".join(violations)


def test_representative_imports_do_not_load_workflow_modules():
    code = """
import importlib, sys
for name in ('gdpx.structures.builders', 'gdpx.structures.regions', 'gdpx.execution.factory', 'gdpx.execution.lifecycle'):
    importlib.import_module(name)
loaded = sorted(name for name in sys.modules if name == 'gdpx.workflow.nodes' or name.startswith('gdpx.workflow.nodes.') or name == 'gdpx.workflow.session' or name.startswith('gdpx.workflow.session.'))
assert not loaded, loaded
"""
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/gdpx-matplotlib")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, check=True)


def test_domain_modules_do_not_import_cli_or_global_registry():
    violations = []
    for directory in DOMAIN_DIRECTORIES:
        for path in (ROOT / "src" / "gdpx" / directory).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "gdpx.cli" or node.module.startswith("gdpx.cli."):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.module}")
                    if node.module in {"gdpx.core.register", "gdpx.core.catalog"}:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: legacy global registry")
    assert not violations, "Reverse/application registry dependencies found:\n" + "\n".join(violations)


def test_all_domain_packages_import_without_workflow_or_optional_dependencies():
    package_names = repr(DOMAIN_DIRECTORIES)
    code = f"""
import importlib, sys
for name in {package_names}:
    importlib.import_module('gdpx.' + name)
loaded = sorted(name for name in sys.modules if name == 'gdpx.workflow.nodes' or name.startswith('gdpx.workflow.nodes.') or name == 'gdpx.workflow.session' or name.startswith('gdpx.workflow.session.'))
assert not loaded, loaded
"""
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/gdpx-matplotlib")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, check=True)


def test_region_factory_does_not_mutate_configuration():
    from gdpx.structures.regions.factory import create_region

    config = {"method": "sphere", "origin": [0.0, 0.0, 0.0], "radius": 2.0}
    original = {"method": "sphere", "origin": [0.0, 0.0, 0.0], "radius": 2.0}
    region = create_region(config)
    assert config == original
    assert region.__class__.__name__ == "SphereRegion"


def test_selector_factory_accepts_workflow_free_selection_config():
    from gdpx.workflow.factory import create_selector
    from gdpx.analysis.selectors.interval import IntervalSelector

    config = {"selection": [{"method": "interval", "period": 7}]}
    original = copy.deepcopy(config)

    selector = create_selector(config)

    assert isinstance(selector, IntervalSelector)
    assert selector.period == 7
    assert config == original


def test_internal_absolute_import_graph_is_acyclic():
    source_root = ROOT / "src"
    modules = {}
    for path in (source_root / "gdpx").rglob("*.py"):
        parts = list(path.relative_to(source_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path

    graph = nx.DiGraph()
    graph.add_nodes_from(modules)
    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                target = max(
                    (candidate for candidate in modules if name == candidate or name.startswith(candidate + ".")),
                    key=len,
                    default=None,
                )
                if target:
                    graph.add_edge(module, target)

    cycles = [sorted(component) for component in nx.strongly_connected_components(graph) if len(component) > 1]
    assert not cycles, f"Internal import cycles found: {cycles}"


def test_potential_base_does_not_own_executor_resolution():
    path = ROOT / "src" / "gdpx" / "providers" / "manager_base.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("gdpx.execution"):
                forbidden.append((node.lineno, node.module))
    assert not forbidden, f"Potential manager imports executor implementations: {forbidden}"


def test_providers_do_not_depend_on_execution_implementations():
    # Provider contracts and discovery stay below execution. Integration
    # packages may implement executor adapters using the public execution SDK.
    contract_modules = {
        "capabilities.py", "configuration.py", "errors.py", "manager.py",
        "provider.py", "specs.py", "targets.py",
    }
    violations = []
    for path in (ROOT / "src" / "gdpx" / "providers").rglob("*.py"):
        if path.parent.name != "providers" or path.name not in contract_modules:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("gdpx.execution"):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.module}")
    assert not violations, "Provider-to-execution imports found:\n" + "\n".join(violations)


def test_legacy_package_trees_are_removed():
    legacy_roots = {
        "backend", "bias", "builder", "colvar", "comparator", "computation",
        "compute", "dataloader", "describer", "expedition", "geometry",
        "graph", "group", "nodes", "potential", "reactor", "region",
        "scheduler", "selector", "session", "trainer", "validator", "worker",
    }
    remaining = sorted(root for root in legacy_roots if (ROOT / "src" / "gdpx" / root).exists())
    assert not remaining


def test_global_legacy_factory_modules_are_removed():
    assert not (ROOT / "src" / "gdpx" / "providers" / "legacy.py").exists()
    assert not (ROOT / "src" / "gdpx" / "providers" / "managed.py").exists()
    assert not (ROOT / "src" / "gdpx" / "execution" / "legacy.py").exists()
    assert not (ROOT / "src" / "gdpx" / "execution" / "compat.py").exists()
    assert not (ROOT / "src" / "gdpx" / "providers" / "compat_registry.py").exists()
    assert not (ROOT / "src" / "gdpx" / "providers" / "trainer_registry.py").exists()
    assert not (ROOT / "src" / "gdpx" / "core" / "register.py").exists()
    assert not (ROOT / "src" / "gdpx" / "core" / "catalog.py").exists()


def test_lazy_registry_defers_module_import():
    from gdpx.core.registry import Registry

    registry = Registry("test")
    registry.register_lazy("counter", "collections", "Counter")
    assert registry["counter"].__name__ == "Counter"
