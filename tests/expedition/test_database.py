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
