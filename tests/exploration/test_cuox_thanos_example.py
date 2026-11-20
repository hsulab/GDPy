"""The bundled CuOx recipe rejects candidates with short O–O contacts."""
from pathlib import Path

import yaml
from ase import Atoms

from gdpx.exploration.persist.thanos import dispatch_thanos


EXAMPLES = Path(__file__).resolve().parents[2] / 'examples/global_optimisation'


def cluster(oo=False):
    atoms = Atoms('Cu4O4', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
                                     [0, 5, 0], [3, 5, 0], [6, 5, 0], [9, 5, 0]])
    if oo:
        atoms.positions[5] = atoms.positions[4] + [1.3, 0, 0]
    return atoms


def test_demo_thanos_checks_oo_even_with_shared_tags():
    config = yaml.safe_load((EXAMPLES / 'basin_hopping/cu4o4_thanos.yaml').read_text())
    thanos = config['system']['thanos']
    assert thanos['restraints'][0]['distance']['max'] == 1.6
    extinct = dispatch_thanos(**thanos)
    assert extinct(cluster()) == 0
    assert extinct(cluster(oo=True)) == 1
    assert config['system']['builders']['init']['method'] == 'random_structure_improved'
