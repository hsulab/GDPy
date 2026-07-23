"""Regression tests for the domain/workflow dependency boundary."""

import ast
import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).parents[1]
DOMAIN_DIRECTORIES = ("builder", "worker", "data", "expedition", "compute", "factory")


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


def test_region_factory_does_not_mutate_configuration():
    from gdpx.factory.region import create_region

    config = {"method": "sphere", "origin": [0.0, 0.0, 0.0], "radius": 2.0}
    original = {"method": "sphere", "origin": [0.0, 0.0, 0.0], "radius": 2.0}
    region = create_region(config)
    assert config == original
    assert region.__class__.__name__ == "SphereRegion"
