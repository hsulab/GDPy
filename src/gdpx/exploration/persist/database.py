import copy
import pathlib
from typing import Optional

import ase.db
from ase import Atoms

CANDIDATES_DATABASE_FILENAME = "candidates.db"

from ..generation import GenerationInfo, GenerationState


def split_description(desc: str) -> tuple[str, str]:
    """Split a description."""
    d = desc.split(":")
    assert len(d) == 2, desc

    return d[0], d[1]


class GlobalOptimisationDatabase:
    """Class to handle the global optimisation database operations.

    The database stores extra information used in tracking evolution.

    """

    def __init__(self, database_fpath: pathlib.Path) -> None:
        """"""
        pathlib.Path(database_fpath).parent.mkdir(parents=True, exist_ok=True)
        self.connection = ase.db.connect(database_fpath)
        self.connection.count()  # Initialise ASE's metadata cache, including an empty database.

        return

    def init_task(self, substrate: Atoms, data: dict[str, int]) -> None:
        """"""
        # Accept the legacy production-size field when opening older task definitions.
        if (
            not ({"generation_size", "population_size"} & data.keys())
            or "initial_population_size" not in data
            or "num_atoms_substrate" not in data
        ):
            raise RuntimeError(
                "Data dictionary must contain 'generation_size', 'initial_population_size', and 'num_atoms_substrate' keys."
            )

        self.connection.write(
            substrate,
            data=data,
            substrate=True,
        )

        return

    def get_param(self, parameter: str) -> Optional[int]:
        """Get a parameter saved when creating the database."""
        param = None
        if self.connection.count() and self.connection.get(1).get("data"):
            param = self.connection.get(1).data.get(parameter, None)

        return param

    def configure_generations(self, initial_size, generation_size, use_extinct=False):
        """Persist sizes without adding a substrate row or changing candidate IDs."""
        metadata = dict(self.connection.metadata)
        settings = dict(initial_size=initial_size, generation_size=generation_size, use_extinct=use_extinct)
        previous = metadata.get("generation_settings")
        if previous is not None and previous != settings:
            raise ValueError("Generation sizes or extinction policy changed; resume with the original recipe.")
        metadata["generation_settings"] = settings
        self.connection.metadata = metadata

    def get_generation_plan(self, generation: int) -> Optional[dict]:
        """Read algorithm-specific construction state, including old GA checkpoints."""
        plans = self.connection.metadata.get("generation_plans", {})
        if str(generation) in plans:
            return copy.deepcopy(plans[str(generation)])
        if self.connection.count():
            data = self.connection.get(1).data or {}
            return copy.deepcopy(data.get("generation_plans", {}).get(str(generation)))
        return None

    def set_generation_plan(self, generation: int, plan: dict) -> None:
        """Persist production state separately from committed evaluation results."""
        metadata = dict(self.connection.metadata)
        plans = copy.deepcopy(metadata.get("generation_plans", {}))
        plans[str(generation)] = copy.deepcopy(plan)
        metadata["generation_plans"] = plans
        self.connection.metadata = metadata

    def generation_candidates(self, generation: int) -> list[Atoms]:
        """Reload the original evaluation batch, including queued/committed inputs."""
        rows = {}
        for row in self.connection.select(relaxed=0, generation=generation):
            if row.formula:
                rows[row.confid] = row
        candidates = []
        for confid, row in sorted(rows.items()):
            atoms = self.connection.get_atoms(row.id, add_additional_information=True)
            atoms.info["confid"] = confid
            candidates.append(atoms)
        return candidates

    def get_substrate(self):
        """Get the substrate."""
        return self.connection.get_atoms(substrate=True)

    def add_unrelaxed_candidate(self, candidate: Atoms, description: str = "", **kwargs):
        """"""
        if description:
            t, desc = split_description(description)
            kwargs.update(**{t: 1, "description": desc})

        key_value_pairs = candidate.info.get("key_value_pairs", {})
        data = candidate.info.get("data", {})

        confid = self.connection.write(
            candidate,
            key_value_pairs=key_value_pairs,
            data=data,
            relaxed=0,
            queued=0,
            extinct=0,
            **kwargs,
        )
        self.connection.update(confid, confid=confid)
        candidate.info["confid"] = confid
        candidate.info["data"] = data
        candidate.info["key_value_pairs"] = {
            **key_value_pairs,
            "extinct": 0,
            **kwargs,
        }

        return

    def add_unrelaxed_step(self, candidate: Atoms, description: str = "", **kwargs) -> None:
        """"""
        confid = candidate.info["confid"]
        if description:
            t, desc = split_description(description)
            kwargs.update(**{t: 1, "description": desc})

        key_value_pairs = candidate.info.get("key_value_pairs", {})
        data = candidate.info.get("data", {})

        self.connection.write(
            candidate,
            key_value_pairs=key_value_pairs,
            data=data,
            confid=confid,
            relaxed=0,
            extinct=0,
            **kwargs,
        )

        return

    def add_relaxed_step(self, atoms: Atoms) -> None:
        """"""
        assert "raw_score" in atoms.info["key_value_pairs"]

        # We may have several entries due to add_unrelaxed_step
        confid = atoms.info["confid"]
        existing = list(self.connection.select(confid=confid, relaxed=1))
        if existing:
            atoms.info["relax_id"] = existing[-1].id
            return

        relax_id = self.connection.write(
            atoms,
            relaxed=1,
            confid=confid,
            key_value_pairs=atoms.info["key_value_pairs"],
            data=atoms.info["data"],
        )
        atoms.info["relax_id"] = relax_id

        return

    def get_one_candidate_by_confid(self, confid: int, add_info: bool = True, mark_as_queued: bool = False) -> Atoms:
        """"""
        images = list(self.connection.select(confid=confid))
        images.sort(key=lambda x: x.mtime)

        # TODO: if there is no images?
        candidate = self.connection.get_atoms(images[-1].id, add_additional_information=add_info)
        if mark_as_queued:
            self.connection.update(id=images[-1].id, queued=1)

        return candidate

    def get_all_relaxed_candidates(self, use_extinct: bool = False):
        """"""
        if use_extinct:
            rows = self.connection.select("relaxed=1,extinct=0", sort="-raw_score")
        else:
            rows = self.connection.select("relaxed=1", sort="-raw_score")

        candidates = []
        for row in rows:
            candidate = self.connection.get_atoms(id=row.id, add_additional_information=True)
            candidate.info["confid"] = row.confid
            candidates.append(candidate)

        return candidates

    def get_number_of_relaxed_candidates(self) -> int:
        """"""
        confids = self._get_all_relaxed_confids()

        return len(confids)

    def _get_all_relaxed_confids(self) -> list[int]:
        """"""
        relaxed_confids = {row.confid for row in self.connection.select(relaxed=1)}

        confids = [confid for confid in relaxed_confids]

        return confids

    def get_all_unrelaxed_candidates(self, mark_as_queued: bool = False) -> list[Atoms]:
        """"""
        confids = self._get_all_unrelaxed_confids()

        candidates = []
        for confid in confids:
            candidate = self.get_one_candidate_by_confid(confid, mark_as_queued=mark_as_queued)
            candidate.info["confid"] = confid
            if "data" not in candidate.info:
                candidate.info["data"] = {}
            candidates.append(candidate)

        return candidates

    def get_number_of_unrelaxed_candidates(self) -> int:
        """"""
        confids = self._get_all_unrelaxed_confids()

        return len(confids)

    def _get_all_unrelaxed_confids(self) -> list[int]:
        """"""
        unrelaxed_confids = {row.confid for row in self.connection.select(relaxed=0)}
        relaxed_confids = {row.confid for row in self.connection.select(relaxed=1)}
        queued_confids = {row.confid for row in self.connection.select(queued=1)}

        confids = [
            confid for confid in unrelaxed_confids if (confid not in relaxed_confids and confid not in queued_confids)
        ]

        return confids

    def mark_as_queued(self, candidate: Atoms) -> None:
        """"""
        confid = candidate.info["confid"]

        rows = list(self.connection.select(f"confid={confid},queued=1"))
        already_queued = len(rows) > 0

        if not already_queued:
            self.connection.write(
                None,
                confid=confid,
                queued=1,
            )

        return

    def get_generation_number(self) -> int:
        """Return the earliest generation whose results are not fully committed."""
        return self.get_generation_info().num

    def get_generation_info(self, generation: Optional[int] = None) -> GenerationInfo:
        """Use unique candidate IDs, never result-file existence, for progress."""
        settings = self.connection.metadata.get("generation_settings")
        if settings is None:
            settings = dict(
                initial_size=self.get_param("initial_population_size"),
                generation_size=self.get_param("generation_size") or self.get_param("population_size"),
                use_extinct=True,
            )
        if not settings["initial_size"] or not settings["generation_size"]:
            raise ValueError("Generation sizes are missing; configure the population before reading progress.")
        produced, relaxed = {}, {}
        survivors = set()
        for row in self.connection.select():
            if "generation" not in row or "relaxed" not in row:
                continue
            num = row.generation
            confid = row.get("confid", row.id)
            (relaxed if row.relaxed else produced).setdefault(num, set()).add(confid)
            if row.relaxed and row.get("extinct", 0) == 0:
                survivors.add(confid)

        def info(num):
            evaluated = relaxed.get(num, set())
            submitted = produced.get(num, set())
            target = settings["initial_size"] if num == 0 else settings["generation_size"]
            if len(evaluated) > target:
                raise RuntimeError(f"Generation {num} has more evaluated candidates than its configured size.")
            plan = self.get_generation_plan(num)
            complete = len(evaluated) == target and (plan is None or plan.get("stage") == "complete")
            state = GenerationState.END_OF_GEN if complete else (
                GenerationState.MID_OF_GEN if submitted or evaluated or plan else GenerationState.BEG_OF_GEN
            )
            return GenerationInfo(num, state, sorted(submitted - evaluated), sorted(evaluated))

        if generation is not None:
            return info(generation)
        num = 0
        while info(num).state is GenerationState.END_OF_GEN:
            num += 1
        current = info(num)
        # Extinction is terminal only between generations, never during ingestion.
        if num > 0 and not produced.get(num) and not relaxed.get(num) and settings["use_extinct"] and not survivors:
            return GenerationInfo(num, GenerationState.EXTINCTED, [], [])
        return current

    def get_participation_in_pairing(self) -> tuple[dict[int, int], list[tuple[int, int]]]:
        """Get how many times each candidate has participated in pairing.

        Note:
            doi.org/10.1021/ja305004a

        """
        entries = self.connection.select(pairing=1)

        frequency = {}
        pairs = []
        for e in entries:
            c1, c2 = e.data["parents"]
            pairs.append(tuple(sorted([c1, c2])))
            if c1 not in frequency.keys():
                frequency[c1] = 0
            frequency[c1] += 1
            if c2 not in frequency.keys():
                frequency[c2] = 0
            frequency[c2] += 1

        return (frequency, pairs)
