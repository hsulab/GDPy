"""Rebuild human-readable GA histories from committed database records."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from ase import Atoms
from ase.db.core import T2000, YEAR

from gdpx.utils.atoms_tags import get_tags_per_species


def _write_atomic(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write('\n'.join(lines) + '\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _timestamp(row):
    # ASE creation times are years since 2000; use the original commit time,
    # rather than changing timestamps whenever the files are regenerated.
    return datetime.fromtimestamp(T2000 + row.ctime * YEAR).strftime('%Y%b%d-%H:%M:%S')


def write_generation_history(database, directory, generation):
    """Include intermediate mutations, whose rows may omit generation/parents."""
    rows = list(database.connection.select(
        relaxed=0, sort='id', columns=['id', 'ctime', 'numbers', 'key_value_pairs', 'data']))
    origins = {}
    for row in rows:
        if row.get('confid') is not None and row.natoms:
            origins.setdefault(row.confid, row)
    plan = database.get_generation_plan(generation) or {}
    lines = [f'# GA generation {generation}: committed candidate production history',
             '# Includes stored reproduction/mutation steps and initial/completion builders.',
             '# Attempt totals include unsuccessful attempts; per-attempt failure diagnostics are not stored.',
             f'# stage: {plan.get("stage", "unknown")} | '
             f'reproduction attempts: {plan.get("reproduction_attempts", 0)} | '
             f'mutation attempts: {plan.get("mutation_attempts", 0)}']
    for row in rows:
        initial = origins.get(row.get('confid'))
        if initial is None or not row.natoms or initial.get('generation', 0) != generation:
            continue
        origin = row.get('origin', initial.get('origin', 'unknown'))
        if row.get('mutation') or 'Mutation' in origin:
            operation = 'mutation'
        elif row.get('pairing') or 'Pairing' in origin or origin == 'Parthenogenesis':
            operation = 'reproduction'
        elif origin.startswith(('InitialBuilder:', 'CompletionBuilder:')):
            operation = 'random'
        else:
            operation = 'unknown'
        parents = row.data.get('parents', initial.data.get('parents', []))
        builder = row.data.get('builder', initial.data.get('builder'))
        lines.append(
            f'{_timestamp(row)} - INFO: row {row.id} confid {row.confid} '
            f'operation={operation} parents={json.dumps(list(parents))} '
            f'origin={origin} builder={builder or "none"} '
            f'description={json.dumps(row.get("description", ""))}')
    _write_atomic(Path(directory) / f'gen{generation}' / 'history.log', lines)


def write_candidate_results(database, directory, preserve_fragments):
    """One row per evaluated candidate, grouped by generation, including extincts."""
    candidates = {}
    for row in database.connection.select(
            relaxed=1, sort='id', columns=['id', 'ctime', 'numbers', 'tags', 'key_value_pairs']):
        candidates[row.confid] = row
    generations = defaultdict(list)
    for row in candidates.values():
        generations[row.get('generation', 0)].append(row)
    lines = ['# GA candidate results: fitness is raw_score (larger is better).',
             '# Timestamps are database creation times; indices restart in each generation.']
    for generation, rows in sorted(generations.items()):
        lines.append(f'# Generation {generation}')
        for index, row in enumerate(sorted(rows, key=lambda row: row.confid)):
            atoms = Atoms(numbers=row.numbers, tags=row.get('tags'))
            if preserve_fragments:
                species = {name: len(groups) for name, groups in get_tags_per_species(atoms).items()}
            else:
                species = Counter(atoms.get_chemical_symbols())
            identities = ' '.join(f'{name}: {count}' for name, count in species.items())
            lines.append(
                f'{_timestamp(row)} - INFO: {index:>4d} confid {row.confid:<6d} '
                f'fitness {row.raw_score:>16.4f} extinct {row.get("extinct", 0):<2d}  {identities}')
    _write_atomic(Path(directory) / 'candidates.log', lines)


def write_search_files(database, directory, calculation_directory, preserve_fragments):
    write_candidate_results(database, directory, preserve_fragments)
    generations = {row.generation for row in database.connection.select(
        'generation', columns=['id', 'key_value_pairs'])}
    # Plans can exist even when production failed before the first candidate.
    generations.update(int(key) for key in database.connection.metadata.get('generation_plans', {}))
    for generation in sorted(generations):
        write_generation_history(database, calculation_directory, generation)
