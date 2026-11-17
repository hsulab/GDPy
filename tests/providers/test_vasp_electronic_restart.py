import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp

from gdpx.providers.vasp.driver import VaspDriver
from gdpx.providers.vasp import restart


@pytest.mark.parametrize("wave,charge,expected", [
    (False, False, (0, 2)), (True, False, (1, 0)),
    (False, True, (0, 1)), (True, True, (1, 1)),
])
def test_checkpoint_selection_and_copy(tmp_path, wave, charge, expected):
    previous = tmp_path / "0000.run"
    previous.mkdir()
    for name, exists in (("WAVECAR", wave), ("CHGCAR", charge)):
        (previous / name).write_bytes(b"checkpoint" if exists else b"")
    calc = Vasp(istart=0, icharg=2)
    restart.restore_electronic_checkpoint(calc, previous, tmp_path)
    assert (calc.int_params["istart"], calc.int_params["icharg"]) == expected
    for name, exists in (("WAVECAR", wave), ("CHGCAR", charge)):
        assert (tmp_path / name).exists() == exists
        if exists:
            assert (tmp_path / name).read_bytes() == (previous / name).read_bytes()


def test_restart_driver_preserves_output_policy_and_restores_geometry(tmp_path, monkeypatch):
    previous = tmp_path / "0000.run"
    previous.mkdir()
    (previous / "INCAR").write_text("ISTART = 0\nLCHARG = .FALSE.\nLWAVE = .FALSE.\n")
    (previous / "CONTCAR").write_text("previous geometry")
    (previous / "WAVECAR").write_bytes(b"wave")
    (previous / "CHGCAR").write_bytes(b"density")
    (previous / "STOPCAR").write_text("LSTOP = .TRUE.\n")
    calc = Vasp(lwave=True, lcharg=True)
    driver = VaspDriver(calc, {"task": "min", "steps": 20}, directory=tmp_path)
    monkeypatch.delenv("GDPX_VASP_STOP_MARGIN_SECONDS", raising=False)
    captured = {}
    def write_input(atoms):
        captured.update(istart=calc.int_params["istart"], icharg=calc.int_params["icharg"],
                        lwave=calc.bool_params["lwave"], lcharg=calc.bool_params["lcharg"])
    with patch.object(calc, "write_input", side_effect=write_input), patch(
        "gdpx.providers.vasp.driver.run_ase_calculator"
    ) as run:
        driver._irun(Atoms("H"), ckpt_wdir=previous, cache_traj=[Atoms("H")])
    assert captured == dict(istart=1, icharg=1, lwave=True, lcharg=True)
    assert (tmp_path / "POSCAR").read_text() == "previous geometry"
    assert not (tmp_path / "STOPCAR").exists()
    run.assert_called_once()


def test_graceful_stop_writes_request_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(time.time() + 10.1))
    with restart.graceful_stop(tmp_path, enabled=True, margin_seconds=10):
        deadline = time.monotonic() + 3
        while not (tmp_path / "STOPCAR").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "STOPCAR").read_text() == "LSTOP = .TRUE.\n"
    assert not (tmp_path / "STOPCAR").exists()


def test_early_exit_cancels_timer_even_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(time.time() + 3600))
    with pytest.raises(ValueError):
        with restart.graceful_stop(tmp_path, enabled=True, margin_seconds=10):
            raise ValueError("calculator failed")
    assert not (tmp_path / "STOPCAR").exists()


def test_disabled_does_not_query_slurm(tmp_path, monkeypatch):
    monkeypatch.setenv("GDPX_VASP_STOP_MARGIN_SECONDS", "7200")
    with patch.object(restart, "slurm_end_time", side_effect=AssertionError):
        with restart.graceful_stop(tmp_path):
            pass


def test_existing_stop_request_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(time.time() + 100))
    (tmp_path / "STOPCAR").write_text("LABORT = .TRUE.")
    with pytest.raises(RuntimeError, match="Existing STOPCAR"):
        with restart.graceful_stop(tmp_path, enabled=True, margin_seconds=10):
            pass
    assert (tmp_path / "STOPCAR").read_text() == "LABORT = .TRUE."


def test_slurm_deadline_fallback(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with patch.object(restart.subprocess, "run", return_value=Mock(
        stdout="JobId=123 EndTime=2026-11-17T18:00:00"
    )) as run:
        assert restart.slurm_end_time() > 0
    assert run.call_args.args[0] == ["scontrol", "show", "job", "-o", "123"]


def test_insufficient_time_refuses_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(time.time() + 10))
    with pytest.raises(RuntimeError, match="Insufficient"):
        with restart.graceful_stop(tmp_path, enabled=True, margin_seconds=100):
            pytest.fail("must not launch")

@pytest.mark.parametrize("enabled", [False, True])
def test_structured_input_reaches_timer(tmp_path, enabled):
    from gdpx.providers.configuration import resolve_executor_parameters
    from contextlib import nullcontext
    params = resolve_executor_parameters({
        "setup": {"graceful_stop": {"enabled": enabled, "margin_seconds": 600}},
        "stop": {"steps": 20}
    }, "min")
    calc = Vasp()
    driver = VaspDriver(calc, dict(params, task="min"), directory=tmp_path)
    previous = tmp_path / "0000.run"
    previous.mkdir()
    (previous / "INCAR").write_text("ISTART = 0\n")
    (previous / "CONTCAR").write_text("geometry")
    with patch.object(calc, "write_input"), patch(
        "gdpx.providers.vasp.driver.run_ase_calculator"
    ), patch("gdpx.providers.vasp.driver.graceful_stop", return_value=nullcontext()) as stop:
        driver._irun(Atoms("H"), ckpt_wdir=previous, cache_traj=[Atoms("H")])
    assert stop.call_args.kwargs == {**restart.GRACEFUL_STOP_DEFAULTS, "enabled": enabled, "margin_seconds": 600}



@pytest.mark.parametrize("options", [
    False, "false", None, [],
    {"enabled": "false"},
    {"margin_seconds": 0},
    {"margin_seconds": -1},
    {"margin_seconds": float("nan")},
    {"margin_seconds": float("inf")},
    {"margin_seconds": True},
    {"margin_seconds": "600"},
    {"enabeld": True},
])
def test_invalid_graceful_stop_input(options):
    from gdpx.providers.vasp.driver import VaspDriverSetting
    with pytest.raises(ValueError):
        VaspDriverSetting(task="min", graceful_stop=options)


def test_graceful_stop_rejects_unsupported_task():
    from gdpx.providers.vasp.driver import VaspDriverSetting
    with pytest.raises(ValueError, match="only for VASP min"):
        VaspDriverSetting(task="md", graceful_stop={"enabled": True})


@pytest.mark.parametrize("options,expected", [
    ({}, {"enabled": False, "margin_seconds": 7200}),
    ({"enabled": True}, {"enabled": True, "margin_seconds": 7200}),
    ({"margin_seconds": 600}, {"enabled": False, "margin_seconds": 600}),
])
def test_graceful_stop_block_defaults(options, expected):
    from gdpx.providers.vasp.driver import VaspDriverSetting
    original = options.copy()
    setting = VaspDriverSetting(task="min", graceful_stop=options)
    assert setting.graceful_stop == {**restart.GRACEFUL_STOP_DEFAULTS, **expected}
    assert options == original


def test_graceful_stop_omitted_is_disabled():
    from gdpx.providers.vasp.driver import VaspDriverSetting
    assert VaspDriverSetting(task="min").graceful_stop == restart.GRACEFUL_STOP_DEFAULTS
