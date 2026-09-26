import logging
from types import SimpleNamespace

import pytest
from ase import Atoms

from gdpx import config
from gdpx.exploration.sampling.logging import MoveLog
from gdpx.exploration.basin_hopping.output import report_setup, bh_logging


def operator():
    return SimpleNamespace(name='move', _print=lambda text: None, _debug=lambda text: None,
                           as_dict=lambda: dict(method='move', particles=['Cu'], temperature=300, max_disp=.1))


@pytest.mark.parametrize('debug', [False, True])
def test_move_logger_context_levels_cleanup_and_ownership(tmp_path, monkeypatch, debug):
    op = operator()
    before = (op._print, op._debug)
    def forbid(*args, **kwargs):
        pytest.fail('logging accessed structures')
    monkeypatch.setattr(Atoms, 'copy', forbid)
    monkeypatch.setattr(Atoms, 'get_potential_energy', forbid)
    old_level = config.logger.level
    config.logger.setLevel(logging.DEBUG if debug else logging.INFO)
    try:
        log = MoveLog(tmp_path / 'moves.log', 1, [op], [1.])
        with pytest.raises(RuntimeError, match='failed'):
            with log:
                with log.operator(op, round=2, chain=3, segment=1, parent=8, operator='0:move'):
                    op._print('first\nsecond')
                    op._debug('details')
                    raise RuntimeError('failed')
        assert (op._print, op._debug) == before
        assert log.handler.stream is None and not log.logger.handlers
        lines = log.path.read_text().splitlines()
        assert all(' - ' in line and 'generation=1' in line for line in lines)
        for label in ('first', 'second'):
            assert any('round=2 chain=3 segment=1 parent=8 operator=0:move' in line and line.endswith(label) for line in lines)
        assert ('details' in '\n'.join(lines)) is debug
        assert 'proposal failed: RuntimeError: failed' in '\n'.join(lines)
        assert 'invocation failed: RuntimeError: failed' in '\n'.join(lines)
    finally:
        config.logger.setLevel(old_level)


def test_operator_summary_survives_quiet_logging(caplog, tmp_path):
    old_level = config.logger.level
    config.logger.setLevel(logging.INFO)
    config.logger.addHandler(caplog.handler)
    try:
        with bh_logging():
            report_setup([operator(), operator()], [.25, .75])
        text = caplog.text
        assert 'operator 0: move | probability: 0.25' in text
        assert 'operator 1: move | probability: 0.75' in text
        assert 'temperature [K]: 300' in text and 'max_disp' in text
        assert 'selection replacement:' not in text
        assert 'move logs:' not in text
    finally:
        config.logger.removeHandler(caplog.handler)
        config.logger.setLevel(old_level)
