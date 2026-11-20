import sys
from types import SimpleNamespace

import pytest

from gdpx import bootstrap, providers
from gdpx.providers.lammps.execution.driver import LmpDriver


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_bootstrap_reports_only_actual_failures(monkeypatch, missing, disabled):
    messages = []
    monkeypatch.setattr(providers, "get_provider_manager", lambda: None)
    monkeypatch.setattr(bootstrap, "WORKFLOW_MODULES", ("example",))
    monkeypatch.setattr(bootstrap.config, "_print", messages.append)

    def import_module(name):
        if missing:
            raise ModuleNotFoundError("No module named 'optional_dep'", name="optional_dep")

    monkeypatch.setattr(bootstrap.importlib, "import_module", import_module)
    bootstrap.bootstrap_registries(disable_import_info=disabled)

    if missing and not disabled:
        assert len(messages) == 2
        assert "FAILED TO IMPORT OPTIONAL MODULES" in messages[0]
        assert "gdpx.workflow.nodes.example" in messages[1]
        assert "optional_dep" in messages[1]
    else:
        assert messages == []


def test_lammps_without_plumed_is_quiet(monkeypatch, capsys):
    for name in tuple(sys.modules):
        if name.startswith("gdpx.providers.plumed.calculators"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "plumed", None)
    driver = object.__new__(LmpDriver)
    calc = SimpleNamespace(units="metal")

    result, params = driver._canonicalise_calculator(calc, {})

    assert result is calc
    assert params["units"] == "metal"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
