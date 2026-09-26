"""On-demand BH trajectory export from committed events and candidate history."""
import json
from pathlib import Path

import ase.db
from ase.io import write

from ..checkpoint import load_data


def export_trajectories(rounds_directory, output_directory, *, database=None):
    """Write one XYZ per chain and return its path.

    ``rounds_directory`` is ``tmp_folder/genN/rounds``. The candidate database
    is discovered in the exploration directory unless explicitly supplied.
    Standalone round coordinators instead use their local ``history.db``.
    Existing exported chain files are overwritten, never appended across calls.
    Only committed events are exported, including after checkpoint cleanup.
    """
    directory, target = Path(rounds_directory), Path(output_directory)
    if (directory / 'final.json').exists():
        state = load_data(directory / 'final.json')
    else:
        manifest = json.loads((directory / 'current.json').read_text())
        state = load_data(directory / manifest['snapshots'][0] / 'state.json')
    if state.get('version') != 5:
        raise ValueError('Unsupported BH event history version.')
    offset = state['context']['journal_offset']
    journal = directory / 'events.jsonl'
    if journal.stat().st_size < offset:
        raise ValueError('Incomplete BH event journal.')
    standalone = database is None and (directory / 'history.db').exists()
    database = Path(database) if database is not None else (
        directory / 'history.db' if standalone else directory.parents[2] / 'candidates.db')
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = ase.db.connect(database)
    target.mkdir(parents=True, exist_ok=True)
    paths = [target / f'mc-{index:04d}.xyz' for index in range(state['count'])]
    written = set()
    with journal.open('rb') as stream:
        while stream.tell() < offset:
            event = json.loads(stream.readline())
            for index, path in enumerate(paths):
                decision = event['decisions'][index]
                if event['step'] and decision not in (0, 4):
                    continue
                identifier = event['candidates'][index]
                row = (connection.get(identifier) if standalone else
                       max(connection.select(confid=identifier), key=lambda row: row.mtime))
                frame = row.toatoms(add_additional_information=True)
                frame.info.update(mcstep=event['step'], segment=event['segments'][index],
                                  source_confid=event['sources'][index],
                                  event='start' if not event['step'] else
                                  ('restart' if decision == 4 else 'accepted'))
                write(path, frame, append=index in written)
                written.add(index)
    return paths
