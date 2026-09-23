"""The CuOx example must verify extinction, not merely contain a Thanos recipe."""
import json
from pathlib import Path
import runpy

import pytest
import yaml
from ase import Atoms
from ase.db import connect
from ase.io import read

from gdpx.exploration.persist.thanos import dispatch_thanos


EXAMPLES = Path(__file__).resolve().parents[2] / 'examples/global_optimisation'
DEMO = runpy.run_path(str(EXAMPLES / 'verify_cu4o4_bh_thanos.py'))


def cluster(oo=False):
    atoms = Atoms('Cu4O4', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
                                     [0, 5, 0], [3, 5, 0], [6, 5, 0], [9, 5, 0]])
    if oo:
        atoms.positions[5] = atoms.positions[4] + [1.3, 0, 0]
    return atoms


def test_demo_thanos_checks_oo_even_with_shared_tags():
    config = yaml.safe_load((EXAMPLES / 'explorations/basin_hopping/cu4o4_thanos.yaml').read_text())
    thanos = config['population']['thanos']
    assert thanos['restraints'][0]['distance']['max'] == DEMO['OO_CUTOFF']
    extinct = dispatch_thanos(**thanos)
    assert extinct(cluster()) == 0
    assert extinct(cluster(oo=True)) == 1
    assert config['population']['builders']['random']['method'] == 'random_structure_improved'


@pytest.mark.parametrize('accepted,extinct,continued,success', [
    (True, 1, True, True), (False, 1, True, False),
    (True, 0, True, False), (True, 1, False, False),
])
def test_demo_verification_requires_complete_extinction_sequence(tmp_path, accepted, extinct, continued, success):
    db = connect(tmp_path / 'candidates.db')
    db.write(cluster(), confid=1, relaxed=1, generation=0, extinct=0)
    db.write(cluster(oo=True), confid=2, relaxed=1, generation=1, extinct=extinct,
             data=dict(accepted=accepted, outcome='extinct' if accepted else 'rejected',
                       round=1, chain=0, segment=0, start_parent=1, parents=[1]))
    if continued:
        db.write(cluster(), confid=3, relaxed=1, generation=1, extinct=0,
                 data=dict(accepted=True, outcome='accepted', round=2, chain=0,
                           segment=1, start_parent=1, parents=[1]))
    rounds = tmp_path / 'tmp_folder/gen1/rounds'
    rounds.mkdir(parents=True)
    (rounds / 'events.jsonl').write_text(json.dumps(dict(step=1, decisions=[4], candidates=[1],
                                                        segments=[1], sources=[1])) + '\n')
    if not success:
        with pytest.raises(RuntimeError):
            DEMO['verify'](tmp_path)
        assert not (tmp_path / 'results/thanos_verification.json').exists()
    else:
        report = DEMO['verify'](tmp_path)
        assert report['verified_restarts'][0]['confid'] == 2
        assert report['verified_restarts'][0]['replacement_confid'] == 1
        assert DEMO['minimum_oo_distance'](read(tmp_path / 'results/oo_extinct.xyz')) == pytest.approx(1.3)
