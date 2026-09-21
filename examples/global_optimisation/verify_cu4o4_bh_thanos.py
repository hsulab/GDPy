"""Verify and export actual O–O extinction events from the Cu4O4 BH demo."""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.io import write


OO_CUTOFF = 1.6  # Angstrom; matches the example's all_atoms Thanos restraint.


def minimum_oo_distance(atoms):
    oxygen = np.flatnonzero(atoms.numbers == 8)
    return min((atoms.get_distance(int(i), int(j), mic=True)
                for offset, i in enumerate(oxygen) for j in oxygen[offset + 1:]),
               default=float('inf'))


def verify(directory):
    """Require a real relaxed O–O trial, acceptance, extinction, and restart."""
    directory = Path(directory)
    database_path = directory / 'candidates.db'
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    rows = list(connect(database_path).select(relaxed=1))
    candidates = {row.confid: row for row in rows}
    events = {event['step']: event for event in (
        json.loads(line) for line in (directory / 'tmp_folder/gen1/rounds/events.jsonl').read_text().splitlines())}
    evidence, frames = [], []
    for row in rows:
        atoms = row.toatoms()
        distance = minimum_oo_distance(atoms)
        if distance >= OO_CUTOFF:
            continue
        if row.extinct != 1:
            raise RuntimeError(f'O–O candidate {row.confid} was not marked extinct.')
        atoms.info.update(confid=row.confid, generation=row.generation,
                          extinct=row.extinct, min_oo_distance=distance)
        frames.append(atoms)
        if row.generation != 1 or not row.data.get('accepted'):
            continue
        if row.data.get('outcome') != 'extinct':
            raise RuntimeError(f'Accepted O–O candidate {row.confid} did not terminate its chain.')
        step, chain = row.data['round'], row.data['chain']
        event = events[step]
        if event['decisions'][chain] != 4:
            continue  # A final-round termination has no remaining moves to restart.
        replacement = candidates[event['candidates'][chain]]
        if replacement.extinct or minimum_oo_distance(replacement.toatoms()) < OO_CUTOFF:
            raise RuntimeError('The replacement candidate violates the O–O rule.')
        segment = row.data['segment'] + 1
        if event['segments'][chain] != segment:
            raise RuntimeError('The replacement did not start a new chain segment.')
        continued = any(other.generation == 1 and other.data.get('chain') == chain
                        and other.data.get('round') == step + 1
                        and other.data.get('segment') == segment
                        and other.data.get('start_parent') == replacement.confid for other in rows)
        if continued:
            evidence.append(dict(confid=row.confid, round=step, chain=chain,
                                 min_oo_distance=distance, replacement_confid=replacement.confid,
                                 next_segment=segment))
    if not evidence:
        raise RuntimeError('No accepted O–O trial followed by extinction, replacement, and continuation was found.')
    output = directory / 'results'
    output.mkdir(exist_ok=True)
    write(output / 'oo_extinct.xyz', frames)
    report = dict(oo_cutoff_angstrom=OO_CUTOFF, extinct_oo_candidates=len(frames),
                  verified_restarts=evidence)
    (output / 'thanos_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='BH expedition directory containing candidates.db')
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
