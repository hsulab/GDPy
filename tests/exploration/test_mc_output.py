"""MC boxes preserve outcome meaning and restore scoped logging on failure."""
import logging
from types import SimpleNamespace

import pytest
from ase import Atoms

from gdpx import config
from gdpx.core.output import Box
from gdpx.execution.monte_carlo.hybrid_monte_carlo import HybridMonteCarlo
from gdpx.execution.monte_carlo.output import mc_box, report_hybrid_intro, report_outcome


@pytest.mark.parametrize("accepted,valid,decision,current", [
    (True, True, "accepted", "-3.000000"),
    (False, True, "rejected", "-2.000000"),
    (False, False, "invalid proposal", "-2.000000"),
    (None, True, "waiting for evaluation", "-2.000000"),
])
def test_outcome_reports_current_energy_without_evaluating_atoms(accepted, valid, decision, current):
    lines = []
    box = Box("step", emit=lines.append)
    # Only len(atoms) is needed; output must not call a calculator or copy atoms.
    result = SimpleNamespace(operator=SimpleNamespace(name="exchange"), diagnostic="Insert_Cu",
                             accepted=accepted, valid=valid, energy=-3., atoms=[None] * 7)
    report_outcome(box, result, -2.)
    box.border("bottom")
    text = "\n".join(lines)
    assert f"decision: {decision}" in text
    assert f"current energy [eV]: {current} | atoms: 7" in text
    assert ("trial -3.000000" in text) is (valid and accepted is not None)
    assert all(len(line) == box.width for line in lines)


@pytest.mark.parametrize("error", [RuntimeError("evaluation failed"), KeyboardInterrupt()])
@pytest.mark.parametrize("debug", [False, True])
def test_box_nesting_failure_and_logging_restoration(caplog, error, debug):
    original_level = config.logger.level
    original_filters = list(config.logger.filters)
    config.logger.setLevel(logging.DEBUG if debug else logging.INFO)
    config.logger.addHandler(caplog.handler)
    try:
        with pytest.raises(type(error)):
            with mc_box("step 1/2") as box:
                config.logger.info("proposal diagnostic")
                config.logger.warning("visible warning")
                worker = Box("worker")
                worker.line("evaluation")
                worker.border("bottom")
                raise error
        assert config.logger.filters == original_filters
        lines = [r.getMessage() for r in caplog.records if getattr(r, "gdpx_panel", False)]
        assert all(len(line) == box.width for line in lines)
        assert lines[-1].startswith("└")
        assert any("│ ┌─ worker" in line for line in lines)
        assert "visible warning" in caplog.text
        assert ("proposal diagnostic" in caplog.text) is debug
        assert ("status: interrupted" if isinstance(error, KeyboardInterrupt) else "status: failed") in caplog.text
        standalone = Box("after")
        standalone.border("bottom")
        assert caplog.records[-2].getMessage().startswith("┌─ after")
    finally:
        config.logger.removeHandler(caplog.handler)
        config.logger.setLevel(original_level)


def test_mc_box_ascii_fallback(caplog, monkeypatch):
    monkeypatch.setattr("gdpx.core.output._unicode_supported", lambda: False)
    config.logger.addHandler(caplog.handler)
    try:
        with mc_box("step 1/1") as box:
            report_outcome(box, SimpleNamespace(operator=SimpleNamespace(name="move"),
                diagnostic="Move_Cu", accepted=True, valid=True, energy=-1., atoms=[None]), 0.)
        for record in caplog.records:
            record.getMessage().encode("ascii")
    finally:
        config.logger.removeHandler(caplog.handler)


def test_hybrid_intro_reports_cycle_and_outputs(caplog):
    cycle = [
        {"provider": "ase", "method": "md", "parameters": {"steps": 20, "temp": 1200}},
        {"provider": "native", "method": "monte_carlo", "parameters": {
            "ensemble": {"method": "nvt", "temperature": 1200}, "stop": {"steps": 5}}},
    ]
    config.logger.addHandler(caplog.handler)
    try:
        report_hybrid_intro(cycle, 10, 1112, "mc.xyz")
    finally:
        config.logger.removeHandler(caplog.handler)
    text = caplog.text
    assert "hybrid monte carlo | intro" in text
    assert "cycle budget: 10 | random seed: 1112" in text
    assert "stage 0: ase/md | 20 MD steps" in text
    assert "stage 1: native/monte_carlo | 5 MC proposals" in text
    assert "outputs: trajectory mc.xyz | MC moves in child opstat.txt files" in text


def test_hybrid_run_routes_progress_to_candidate_log(caplog, tmp_path):
    engine = object.__new__(HybridMonteCarlo)
    engine.directory = tmp_path
    engine.set_rng(seed=17)

    def run_with_worker(atoms):
        config.logger.info("particles in system: Cu 8")
        config.logger.info("succeed to insert after 2 attempts")
        with mc_box("visible") as box:
            box.line("concise progress")

    engine._run = run_with_worker
    engine.read_convergence = lambda: True
    original_level = config.logger.level
    config.logger.setLevel(logging.INFO)
    config.logger.addHandler(caplog.handler)
    try:
        engine.run(Atoms("Cu"))
    finally:
        config.logger.removeHandler(caplog.handler)
        config.logger.setLevel(original_level)
    assert "particles in system" not in caplog.text
    assert "succeed to insert" not in caplog.text
    assert "concise progress" not in caplog.text
    text = (tmp_path / "mc.out").read_text()
    assert "particles in system" in text
    assert "succeed to insert" in text
    assert "concise progress" in text
