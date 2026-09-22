"""GA text exports survive resumes and retain intermediate production steps."""
from ase import Atoms

from gdpx.exploration.genetic_algorithm.files import write_search_files
from gdpx.exploration.persist.database import GlobalOptimisationDatabase


def test_history_preserves_production_steps_and_rebuilds_without_duplicates(tmp_path):
    db = GlobalOptimisationDatabase(tmp_path / 'candidates.db')
    atoms = Atoms('Cu2', tags=[1, 2])
    db.connection.write(atoms, relaxed=0, confid=1, generation=0,
                        origin='InitialBuilder:seed', data={'builder': 'seed'})
    db.connection.write(atoms, relaxed=0, confid=2, generation=1, pairing=1,
                        origin='Pairing', description='1 1', data={'parents': [1, 1]})
    # Intermediate mutation rows need not repeat the generation or parent metadata.
    db.connection.write(atoms, relaxed=0, confid=2, mutation=1,
                        origin='RattleMutation', description='rattle')
    db.connection.write(atoms, relaxed=0, confid=3, generation=1, mutation=1,
                        origin='MutationCandidateUnrelaxed', data={'parents': [1]})
    db.connection.write(atoms, relaxed=0, confid=4, generation=1,
                        origin='CompletionBuilder:random', data={'builder': 'random'})
    db.set_generation_plan(1, {'stage': 'complete', 'reproduction_attempts': 3,
                               'mutation_attempts': 2})
    # Queue markers have no structure and must not appear as production events.
    db.connection.write(Atoms(), relaxed=0, confid=2, queued=1)
    write_search_files(db, tmp_path, tmp_path / 'tmp_folder', use_tags=True)
    path = tmp_path / 'tmp_folder/gen1/history.log'
    history = path.read_text()
    assert 'reproduction attempts: 3 | mutation attempts: 2' in history
    assert history.count(' - INFO: ') == 4
    assert 'operation=reproduction parents=[1, 1]' in history
    assert 'operation=mutation parents=[1, 1]' in history
    assert 'description="rattle"' in history
    assert 'operation=mutation parents=[1]' in history
    assert 'operation=random parents=[] origin=CompletionBuilder:random builder=random' in history
    assert 'InitialBuilder' not in history
    assert 'builder=seed' in (tmp_path / 'tmp_folder/gen0/history.log').read_text()
    write_search_files(db, tmp_path, tmp_path / 'tmp_folder', use_tags=True)
    assert path.read_text() == history


def test_candidate_results_use_committed_fitness_species_and_stable_timestamps(tmp_path):
    db = GlobalOptimisationDatabase(tmp_path / 'candidates.db')
    atoms = Atoms('Cu2', tags=[1, 2])
    db.connection.write(atoms, relaxed=0, confid=5, generation=1, origin='Pairing')
    db.connection.write(atoms, relaxed=1, confid=2, generation=0,
                        raw_score=-10.3166, target=10.3166, extinct=0)
    db.connection.write(atoms, relaxed=1, confid=3, generation=1,
                        raw_score=-10.5847, target=10.5847, extinct=1)
    # A historical duplicate relaxation still produces just one candidate line.
    db.connection.write(atoms, relaxed=1, confid=3, generation=1,
                        raw_score=-10.3699, target=10.3699, extinct=1)
    write_search_files(db, tmp_path, tmp_path / 'tmp_folder', use_tags=True)
    path = tmp_path / 'candidates.log'
    result = path.read_text()
    assert '# Generation 0' in result and '# Generation 1' in result
    assert result.count(' - INFO: ') == 2
    assert 'confid 5' not in result
    assert '-10.3166 extinct 0' in result and '-10.3699 extinct 1' in result
    assert result.count('Cu: 2') == 2
    write_search_files(db, tmp_path, tmp_path / 'tmp_folder', use_tags=True)
    assert path.read_text() == result
    # Untagged searches show atom counts rather than one combined fragment.
    write_search_files(db, tmp_path, tmp_path / 'tmp_folder', use_tags=False)
    assert path.read_text().count('Cu: 2') == 2


def test_empty_failed_generation_still_has_history(tmp_path):
    db = GlobalOptimisationDatabase(tmp_path / 'candidates.db')
    db.set_generation_plan(0, {'stage': 'initial'})
    write_search_files(db, tmp_path, tmp_path / 'tmp_folder', use_tags=False)
    assert '# stage: initial' in (tmp_path / 'tmp_folder/gen0/history.log').read_text()
    assert ' - INFO: ' not in (tmp_path / 'candidates.log').read_text()
