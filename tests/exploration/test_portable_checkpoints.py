"""Portable data must retain scientific state without Python object pickling."""
import json

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLengths
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.exploration.accepted_state import save_accepted_state, load_accepted_state
from gdpx.exploration.checkpoint import save_data, load_data


def test_structure_roundtrip_retains_arrays_constraints_and_results(tmp_path):
    atoms = Atoms('Cu2', positions=[[0., 0., 0.], [2., 0., 0.]], cell=[10]*3,
                  pbc=[True, False, False], tags=[3, 7], momenta=np.arange(6).reshape(2, 3))
    atoms.new_array('custom', np.array([1, 2], dtype='>i4'))
    atoms.set_constraint([FixAtoms(indices=[0]), FixBondLengths([[0, 1]])])
    atoms.info['nested'] = {'tuple': (1, 'x'), 'array': np.array([1+2j]), 'scalar': np.float32(1.25)}
    forces = np.arange(6).reshape(2, 3).astype(float)
    atoms.calc = SinglePointCalculator(atoms, energy=-3.5, forces=forces, stress=np.arange(6.))
    path = tmp_path / 'structure.json'
    save_accepted_state(path, atoms, -3.5)
    restored = load_accepted_state(path)
    for key, value in atoms.arrays.items():
        np.testing.assert_array_equal(restored.arrays[key], value)
        assert restored.arrays[key].dtype == value.dtype
    assert [c.todict()['name'] for c in restored.constraints] == ['FixAtoms', 'FixBondLengths']
    np.testing.assert_array_equal(restored.constraints[0].index, [0])
    np.testing.assert_array_equal(restored.constraints[1].pairs, [[0, 1]])
    np.testing.assert_array_equal(restored.cell, atoms.cell)
    np.testing.assert_array_equal(restored.pbc, atoms.pbc)
    assert restored.info['nested']['tuple'] == (1, 'x')
    assert isinstance(restored.info['nested']['scalar'], np.float32)
    np.testing.assert_array_equal(restored.info['nested']['array'], [1+2j])
    assert restored.get_potential_energy() == -3.5
    np.testing.assert_array_equal(restored.calc.results['forces'], forces)
    assert not list(tmp_path.glob('*.pkl'))
    payload = json.loads(path.read_text())
    with np.load(tmp_path / payload['arrays'], allow_pickle=False) as arrays:
        assert all(not arrays[name].dtype.hasobject for name in arrays.files)


@pytest.mark.parametrize('bit_generator', [np.random.PCG64, np.random.MT19937])
def test_rng_and_nested_metadata_roundtrip(tmp_path, bit_generator):
    rng = np.random.Generator(bit_generator(123))
    data = {'rng': rng.bit_generator.state, ('compound', 1): {'large': 2**128},
            'infinite': float('inf'), '__checkpoint_type__': 'user metadata'}
    save_data(tmp_path / 'data.json', data)
    restored = load_data(tmp_path / 'data.json')
    actual = np.random.Generator(bit_generator())
    actual.bit_generator.state = restored['rng']
    np.testing.assert_array_equal(actual.random(100), rng.random(100))
    assert restored[('compound', 1)]['large'] == 2**128
    assert restored['__checkpoint_type__'] == 'user metadata'
    assert np.isinf(restored['infinite'])


@pytest.mark.parametrize('value', [object(), np.array([{}], dtype=object)])
def test_unsupported_values_never_fall_back_to_pickle(tmp_path, value):
    with pytest.raises(TypeError, match='[Oo]bject|Unsupported'):
        save_data(tmp_path / 'data.json', value)
    assert not (tmp_path / 'data.json').exists()


def test_codec_streams_original_arrays(tmp_path, monkeypatch):
    original = np.arange(100)
    savez = np.savez
    def record(stream, **arrays):
        assert list(arrays.values())[0] is original
        return savez(stream, **arrays)
    monkeypatch.setattr(np, 'savez', record)
    save_data(tmp_path / 'data.json', {'array': original})


def test_corrupt_array_is_detected(tmp_path):
    save_data(tmp_path / 'data.json', np.arange(5))
    payload = json.loads((tmp_path / 'data.json').read_text())
    (tmp_path / payload['arrays']).write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'):
        load_data(tmp_path / 'data.json')
