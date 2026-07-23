"""Regression tests for the domain/workflow dependency boundary."""

import ast
import os
import pathlib
import subprocess
import sys

import networkx as nx


ROOT = pathlib.Path(__file__).parents[1]
DOMAIN_DIRECTORIES = tuple(
    path.name
    for path in (ROOT / "src" / "gdpx").iterdir()
    if path.is_dir() and (path / "__init__.py").exists() and path.name not in {"cli", "nodes", "session"}
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
                    if name == "gdpx.nodes" or name.startswith("gdpx.nodes.") or name == "gdpx.session" or name.startswith("gdpx.session."):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "Domain-to-workflow imports found:\n" + "\n".join(violations)


def test_representative_imports_do_not_load_workflow_modules():
    code = """
import importlib, sys
for name in ('gdpx.builder', 'gdpx.factory.region', 'gdpx.factory.computer', 'gdpx.compute.service'):
    importlib.import_module(name)
loaded = sorted(name for name in sys.modules if name == 'gdpx.nodes' or name.startswith('gdpx.nodes.') or name == 'gdpx.session' or name.startswith('gdpx.session.'))
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
                    if node.module == "gdpx.core.register" and any(alias.name == "registers" for alias in node.names):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: global registers")
    assert not violations, "Reverse/application registry dependencies found:\n" + "\n".join(violations)


def test_all_domain_packages_import_without_workflow_or_optional_dependencies():
    package_names = repr(DOMAIN_DIRECTORIES)
    code = f"""
import importlib, sys
for name in {package_names}:
    importlib.import_module('gdpx.' + name)
loaded = sorted(name for name in sys.modules if name == 'gdpx.nodes' or name.startswith('gdpx.nodes.') or name == 'gdpx.session' or name.startswith('gdpx.session.'))
assert not loaded, loaded
"""
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/gdpx-matplotlib")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, check=True)


def test_region_factory_does_not_mutate_configuration():
    from gdpx.factory.region import create_region

    config = {"method": "sphere", "origin": [0.0, 0.0, 0.0], "radius": 2.0}
    original = {"method": "sphere", "origin": [0.0, 0.0, 0.0], "radius": 2.0}
    region = create_region(config)
    assert config == original
    assert region.__class__.__name__ == "SphereRegion"


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


def test_lazy_registry_defers_module_import():
    from gdpx.core.registry import Registry

    registry = Registry("test")
    registry.register_lazy("counter", "collections", "Counter")
    assert registry["counter"].__name__ == "Counter"
