import collections
import dataclasses
import enum
import pathlib
from typing import Optional

import ase.db
from ase import Atoms

GenerationState = enum.Enum(
    "GenerationState",
    (
        "BEG_OF_GEN",
        "MID_OF_GEN",
        "END_OF_GEN",
        "EXTINCTED",
    ),
)


@dataclasses.dataclass
class GenerationInfo:
    #: The generation number.
    num: int

    #: The generation state.
    state: GenerationState

    #: The unrelaxed candidate confids.
    unrelaxed_confids: list[int]

    #: The relaxed candidate confids.
    relaxed_confids: list[int]

    def __post_init__(self) -> None:
        """"""
        self.num_unrelaxed = len(self.unrelaxed_confids)
        self.num_relaxed = len(self.relaxed_confids)

        return


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
        self.connection = ase.db.connect(database_fpath)

        return

    def init_task(self, substrate: Atoms, data: dict[str, int]) -> None:
        """"""
        # We must have three integers, population_size, initial_population_size, and num_atoms_substrate in data
        if "population_size" not in data or "initial_population_size" not in data or "num_atoms_substrate" not in data:
            raise RuntimeError(
                "Data dictionary must contain 'population_size', 'initial_population_size', and 'num_atoms_substrate' keys."
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
        if self.connection.get(1).get("data"):
            param = self.connection.get(1).data.get(parameter, None)

        return param

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
        # rows = list(self.connection.select(confid=confid, relaxed=0))

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
        key_value_pairs = candidate.info.get("key_value_pairs", {})

        self.connection.write(
            None,
            confid=confid,
            queued=1,
            key_value_pairs=key_value_pairs,
        )

        return

    def get_generation_number(self) -> int:
        """Get the current generation number.

        The population size of the first generation can be different from the following ones.

        Returns:
            int: generation number

        """
        init_pop_size = self.get_param("initial_population_size")
        assert isinstance(init_pop_size, int)
        pop_size = self.get_param("population_size")
        assert isinstance(pop_size, int)

        all_candidates = list(self.connection.select(relaxed=1))
        counter = collections.Counter([c.generation for c in all_candidates])
        generations = sorted(list(counter.keys()))
        num_generations = len(generations)
        if num_generations == 0:
            gen_num = 0
        else:
            if num_generations == 1:
                if counter[0] < init_pop_size:
                    gen_num = 0
                else:
                    assert counter[0] == init_pop_size
                    gen_num = 1
            else:
                gen_num = max(generations)
                if counter[gen_num] < pop_size:
                    ...
                else:
                    assert counter[gen_num] == pop_size
                    gen_num += 1

        return gen_num

    def get_generation_info(self) -> GenerationInfo:
        """Get the current generation state.

        Returns:
            GenerationState: generation state

        """
        gen_num = self.get_generation_number()

        unrelaxed_candidate_rows = list(self.connection.select(f"relaxed=0,generation={gen_num}"))
        unrelaxed_confids = {row.confid for row in unrelaxed_candidate_rows}
        num_unrelaxed = len(unrelaxed_confids)

        relaxed_candidate_rows = list(self.connection.select(f"relaxed=1,generation={gen_num}"))
        relaxed_confids = {row.confid for row in relaxed_candidate_rows}
        num_relaxed = len(relaxed_confids)

        if num_relaxed == 0:
            gen_state = GenerationState.BEG_OF_GEN
        else:
            if num_relaxed < num_unrelaxed:
                gen_state = GenerationState.MID_OF_GEN
            else:
                gen_state = GenerationState.END_OF_GEN

        gen_info = GenerationInfo(
            num=gen_num,
            state=gen_state,
            unrelaxed_confids=list(unrelaxed_confids),
            relaxed_confids=list(relaxed_confids),
        )

        return gen_info

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
