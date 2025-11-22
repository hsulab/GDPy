#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

import ase.db
from ase import Atoms


class GlobalOptimisationDatabase:
    """Class to handle the global optimisation database operations.

    The database stores extra information used in tracking evolution.

    """

    def __init__(self, database_fpath: pathlib.Path) -> None:
        """"""
        self.connection = ase.db.connect(database_fpath)

        return

    def add_unrelaxed_candidate(self, candidate: Atoms, **kwargs):
        """"""
        confid = self.connection.write(candidate, relaxed=0, queued=0, extinct=0, **kwargs)
        self.connection.update(confid, confid=confid)
        candidate.info["confid"] = confid

        return

    def add_relaxed_step(self, atoms: Atoms) -> None:
        """"""
        assert "raw_score" in atoms.info["key_value_pairs"]

        confid = atoms.info["confid"]
        rows = list(self.connection.select(confid=confid, relaxed=0))
        assert len(rows) == 1

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

    def get_number_of_relaxed_candidates(self):
        """"""
        confids = self._get_all_relaxed_confids()

        return len(confids)

    def _get_all_relaxed_confids(self):
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

    def _get_all_unrelaxed_confids(self):
        """"""
        relaxed_confids = {row.confid for row in self.connection.select(relaxed=1)}
        unrelaxed_confids = {row.confid for row in self.connection.select(relaxed=0)}
        queued_confids = {row.confid for row in self.connection.select(queued=1)}

        confids = [
            confid for confid in unrelaxed_confids if (confid not in relaxed_confids and confid not in queued_confids)
        ]

        return confids


if __name__ == "__main__":
    ...
