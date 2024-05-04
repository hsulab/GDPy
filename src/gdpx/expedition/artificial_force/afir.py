#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import dataclasses
import itertools
import json
import pathlib
import pickle
import time
from typing import List, Mapping, NoReturn, Union

import numpy as np
from ase import Atoms
from ase.formula import Formula
from ase.geometry import find_mic
from ase.io import read, write

from .. import (
    AtomsNDArray,
    GridDriverBasedWorker,
    MolecularAdsorbate,
    create_a_group,
    create_a_molecule_group,
    create_mixer,
    find_molecules,
    str2array,
)
from ..expedition import AbstractExpedition


def convert_index_to_formula(atoms, group_indices: List[List[int]]):
    """"""
    formulae = []
    for g in group_indices:
        symbols = [atoms[i].symbol for i in g]
        formulae.append(Formula.from_list(symbols).format("hill"))
    # formulae = sorted(formulae)

    return formulae


def find_target_fragments(
    atoms, target_commands: List[str]
) -> Mapping[str, List[List[int]]]:
    """Find target fragments in the structure to react.

    This is a wrapper for group commands as there are several ways to defind
    a (atomic) group but we need a group of molecules here. Optional ways are
    using `molecule` all, using `tag`...

    """
    fragments = {}  # Mapping[str,List[List[int]]]
    target_molecules = None

    ngroups = len(target_commands)
    if ngroups == 1:
        atomic_indices = create_a_group(atoms, target_commands[0])
        fragments = find_molecules(atoms, atomic_indices)
    else:
        assert ngroups >= 2, "Need at least 2 groups..."
        for group_command in target_commands:
            fragments[group_command] = create_a_molecule_group(atoms, group_command)

    return fragments


def get_last_atoms(frames):
    """"""
    num_frames = len(frames)
    for i in range(num_frames, 0, -1):
        atoms = frames[i - 1]
        if atoms is not None:
            break

    return atoms


@dataclasses.dataclass
class ReactionSpace:

    group: str
    reactions: list[dict] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """"""
        self._reactant_list = [tuple(sorted(p["reactants"])) for p in self.reactions]
        self._distance_list = [p["distance"] for p in self.reactions]

        return

    def get_reactive_indices(self, atoms: Atoms):
        """"""

        return create_a_group(atoms, group_command=self.group)

    def is_reaction_possible(self, atoms, molecules) -> bool:
        """"""
        is_possible = False

        assert len(molecules) == 2
        m0, m1 = molecules[0], molecules[1]

        if m0.atomic_indices != m1.atomic_indices:
            reactants = tuple(sorted([m0.chemical_formula, m1.chemical_formula]))
            if reactants in self._reactant_list:
                com_vec = m0.get_center_of_mass() - m1.get_center_of_mass()
                vec, dis = find_mic(com_vec, atoms.cell)
                if dis < self._distance_list[self._reactant_list.index(reactants)]:
                    is_possible = True
                    print(m0, m1, dis)
        else:
            ...

        return is_possible


class AFIRSearch(AbstractExpedition):

    def __init__(
        self,
        builder,
        reaction_space: dict,
        gamma: Union[float, str] = "2.0",
        *args,
        **kwargs,
    ) -> None:
        """Define some basic parameters for the afir search.

        Args:
            builder: StructureBuilder.

        """
        super().__init__(*args, **kwargs)

        self.builder = builder
        self.reaction_space = ReactionSpace(**reaction_space)

        self.gamma_factors = gamma

        return

    def _spawn_computers(
        self, pair: List[List[int]], gamma_factors: List[float], *args, **kwargs
    ):
        """Spawn AFIR computers."""
        # if hasattr(self.worker.potter, "remove_loaded_models"):
        #     self.worker.potter.remove_loaded_models()

        # get parameters from host worker
        host_dict = self.worker.potter.as_dict()
        # self._print(f"{host_dict =}")

        driver_dict = self.worker.driver.as_dict()
        # self._print(f"{driver_dict =}")

        bias_list = []
        for g in gamma_factors:
            curr_bias = dict(name="bias", params={})
            curr_bias["params"]["backend"] = "ase"
            curr_bias["params"]["method"] = "afir"
            curr_bias["params"]["gamma"] = g
            curr_bias["params"]["groups"] = pair
            curr_bias["params"]["use_pbc"] = False  # FIXME:
            bias_list.append(curr_bias)

        # Use shared host potter to reduce loading time and memory usage
        # as some large models may lead to OOM issue if non-shared.
        potters, drivers = [], []
        for b in bias_list:
            # p = create_mixer(host_dict, b)
            p = create_mixer(self.worker.potter, b)
            driver = p.create_driver(driver_dict)
            potters.append(p)
            drivers.append(driver)

        # for p in potters:
        #     self._print(f"{p.potters[0]}")

        return potters, drivers

    def run(self, *args, **kwargs) -> None:
        """"""
        super().run()

        # assume input structures are minimised
        structures = self.builder.run()
        num_structures = len(structures)
        assert num_structures == 1, "Only one structure for now."

        atoms = structures[0]

        # start expedition
        self._print("---------------------------")
        self._print("| AFIRSearch starts...... |")
        self._print("---------------------------")

        # - find fragments
        reactive_indices = self.reaction_space.get_reactive_indices(atoms)
        self._print(f"{reactive_indices =}")

        fragments = find_molecules(atoms, reactive_indices=reactive_indices)
        self._print(fragments)
        num_fragraments = len(fragments)

        # pair order is random
        pair_data_fpath = self.directory / "pairs.json"
        if not pair_data_fpath.exists():
            possible_pairs = []
            for pp in itertools.combinations(range(num_fragraments), 2):
                m0, m1 = fragments[pp[0]], fragments[pp[1]]
                if self.reaction_space.is_reaction_possible(atoms, [m0, m1]):
                    possible_pairs.append([m0, m1])
            self._print(f"{possible_pairs =}")

            # serialisation pair data
            pair_data = [[m0.as_dict(), m1.as_dict()] for m0, m1 in possible_pairs]
            with open(pair_data_fpath, "w") as fopen:
                json.dump(pair_data, fopen, indent=2)
        else:
            with open(pair_data_fpath, "r") as fopen:
                pair_data = json.load(fopen)
            possible_pairs = [
                [MolecularAdsorbate.from_dict(p0), MolecularAdsorbate.from_dict(p1)]
                for (p0, p1) in pair_data
            ]

        num_pairs = len(possible_pairs)

        # NOTE: Sampled structures are
        #       (num_structures, num_reactions, num_gamma, num_frames)
        #       num_computations = num_structures * num_reactions * num_gamma??

        # TODO: If gamma is too large, the first computation leads to reaction

        comput_dpath = self.directory / "comput"
        comput_dpath.mkdir(exist_ok=True)

        # - run each pair
        grid_workers = []
        for i, rxn_pair in enumerate(possible_pairs):
            # write to log
            self._print(f"===== Pair {i} =====")
            # reactants = convert_index_to_formula(atoms, rxn_pair)
            self._print("reactants: ")
            # for ir, r in enumerate(reactants):
            #     self._print(f"  {r:>24s}_{rxn_pair[ir]}")
            for r in rxn_pair:
                self._print(f"  {str(r):<48s}")
            group_indices = [m.atomic_indices for m in rxn_pair]

            potters, drivers = self._spawn_computers(group_indices, self.gamma_factors)
            curr_worker = GridDriverBasedWorker(potters=potters, drivers=drivers)
            curr_worker.directory = comput_dpath / f"pair{i}"
            curr_worker.batchsize = self.worker.batchsize
            grid_workers.append(curr_worker)

            num_potters = len(potters)
            curr_worker.run([atoms for _ in range(num_potters)])

        # extract each pair
        worker_status = []
        for i, curr_worker in enumerate(grid_workers):
            curr_worker.inspect(resubmit=True)
            if curr_worker.get_number_of_running_jobs() == 0:
                worker_status.append(True)
            else:
                worker_status.append(False)

        if all(worker_status):
            results = self._extract_results(grid_workers)
            self._print(f"{results =}")
            assert num_pairs == results.shape[0]

            # save pathways?
            ret_dir = self.directory / "results"
            ret_dir.mkdir(exist_ok=True)
            for i in range(num_pairs):
                pseudo_pathway = [get_last_atoms(p) for p in results[i]]
                # self._print(f"{pseudo_pathway =}")
                write(ret_dir / f"prp{i}.xyz", pseudo_pathway)

            # dump finished flag
            with open(self.directory / "FINISHED", "w") as fopen:
                fopen.write("")

        return

    def _extract_results(self, workers):
        """"""
        num_workers = len(workers)

        results = []
        for i, worker in enumerate(workers):
            curr_results = worker.retrieve(include_retrieved=True)
            results.append(curr_results)
        results = AtomsNDArray(results)

        return results

    def _postprocess(self):
        # if all(worker_status):
        #     self._print("---------------------------")
        #     self._print("| AFIRSearch is finished. |")
        #     self._print("---------------------------")
        #     nimages = len(gamma_list)
        #     for i, (pair, curr_worker) in enumerate(zip(possible_pairs, grid_workers)):
        #         self._print(f"===== Pair {i} =====")
        #         reactants = convert_index_to_formula(atoms, pair)
        #         self._print("Reactants:")
        #         self._print(reactants)
        #         self._print(pair)
        #         curr_trajs = curr_worker.retrieve(include_retrieved=True)
        #         pseudo_pathway = [t[-1] for t in curr_trajs]
        #         end_atoms = pseudo_pathway[-1]
        #         prod_indices = find_product(end_atoms, pair)
        #         products = convert_index_to_formula(end_atoms, prod_indices)
        #         self._print("Products:")
        #         self._print(products)
        #         if sorted(reactants) == sorted(products):
        #             # TODO: index may change? molecule reconstruct?
        #             self._print("nothing happens...")
        #         else:
        #             self._print("reaction happens...")
        #             # TODO: postprocess, minimise FS to create a pathway
        #     # -
        #     with open(self.directory / "FINISHED", "w") as fopen:
        #         fopen.write(
        #             f"FINISHED AT {time.asctime( time.localtime(time.time()) )}."
        #         )
        # else:
        #     self._print("Some calculations are unfinished.")

        return

    def read_convergence(self, *args, **kwargs):
        """"""
        converged = False
        if (self.directory / "FINISHED").exists():
            converged = True

        return converged

    def get_workers(self, *args, **kwargs):
        """"""
        workers = []
        if hasattr(self.worker.potter, "remove_loaded_models"):
            self.worker.potter.remove_loaded_models()

        with open(self.directory / "pairs.json", "r") as fopen:
            pairs = json.load(fopen)
        num_pairs = len(pairs)

        workers = []
        for i in range(num_pairs):
            potters, drivers = self._spawn_computers(pairs[i], self.gamma_factors)
            curr_worker = GridDriverBasedWorker(potters=potters, drivers=drivers)
            curr_worker.directory = self.directory / "comput" / f"pair{i}"
            assert curr_worker.directory.exists()
            workers.append(curr_worker)

        return workers


if __name__ == "__main__":
    ...
