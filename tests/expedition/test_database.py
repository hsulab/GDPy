import pytest
from ase import Atoms

from gdpx.expedition.persist.database import GlobalOptimisationDatabase


def test_add_unrelaxed_candidate_updates_in_memory_metadata(tmp_path):
    database = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    candidate = Atoms("H")

    database.add_unrelaxed_candidate(
        candidate,
        description="random: OffspringGenerator",
        origin="RandomCandidateUnrelaxed",
        generation=1,
    )

    assert candidate.info["confid"] == 1
    assert candidate.info["data"] == {}
    assert candidate.info["key_value_pairs"] == {
        "extinct": 0,
        "origin": "RandomCandidateUnrelaxed",
        "generation": 1,
        "random": 1,
        "description": " OffspringGenerator",
    }

    candidate.info["key_value_pairs"]["raw_score"] = 0.0
    database.add_relaxed_step(candidate)
    assert len(database.get_all_relaxed_candidates(use_extinct=True)) == 1


def test_queued_pairing_candidate_does_not_create_a_pairing_record(tmp_path):
    database = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    candidate = Atoms("H2")
    candidate.info["data"] = {"parents": [3, 7]}
    candidate.info["key_value_pairs"] = {"origin": "Pairing"}

    database.add_unrelaxed_candidate(candidate, description="pairing: 3 7", generation=1)
    database.mark_as_queued(candidate)

    queued_row = database.connection.get(confid=candidate.info["confid"], queued=1)
    assert dict(queued_row.key_value_pairs) == {
        "confid": candidate.info["confid"],
        "queued": 1,
    }
    assert database.get_participation_in_pairing() == ({3: 1, 7: 1}, [(3, 7)])


def test_pairing_record_without_parents_raises(tmp_path):
    database = GlobalOptimisationDatabase(tmp_path / "candidates.db")
    database.connection.write(None, pairing=1, queued=1)

    with pytest.raises(KeyError, match="parents"):
        database.get_participation_in_pairing()
