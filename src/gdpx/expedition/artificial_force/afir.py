#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import dataclasses
import itertools
import pathlib
import time
import pickle
import json

from typing import NoReturn, Union, List, Mapping

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.formula import Formula

from .. import create_mixer
from .. import AtomsNDArray
from ..expedition import AbstractExpedition

from gdpx.worker.grid import GridDriverBasedWorker
from gdpx.builder.group import create_a_group, create_a_molecule_group
from gdpx.graph.molecule import find_product, find_molecules


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
        atoms = frames[i-1]
        if atoms is not None:
            break

    return atoms


@dataclasses.dataclass
class ReactionSpace:

    ...

    def __post_init__(self):
        """"""

        return


class AFIRSearch(AbstractExpedition):

    def __init__(
        self, builder, reaction: dict, gamma: Union[float, str] = "2.0", *args, **kwargs
    ) -> None:
        """Define some basic parameters for the afir search.

        Args:
            builder: StructureBuilder.

        """
        super().__init__(*args, **kwargs)

        self.builder = builder
        self.reaction = reaction

        self.gamma_factors = gamma

        return

    def _prepare_fragments(self, atoms, *args, **kwargs):
        """"""
        data_path = self.directory / "_data"
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)

        # - find possible reaction pairs
        frag_fpath = data_path / "fragments.pkl"
        # TODO: assure the pair order is the same when restart
        if not frag_fpath.exists():
            fragments = find_target_fragments(atoms, self.target)
            with open(frag_fpath, "wb") as fopen:
                pickle.dump(fragments, fopen)
        else:
            with open(frag_fpath, "rb") as fopen:
                fragments = pickle.load(fopen)

        # TODO: assert molecules in one group are the same type?
        self._print("Found Target Fragments: ")
        for k, v in fragments.items():
            self._print("  {:<24s}:  {}".format(k, v))

        frag_list = []
        for k, v in fragments.items():
            frag_list.append(v)

        ntypes = len(frag_list)
        comb = combinations(range(ntypes), 2)

        possible_pairs = []
        for i, j in comb:
            f1, f2 = frag_list[i], frag_list[j]
            possible_pairs.extend(list(product(f1, f2)))

        with open(self.directory / "_data" / "pairs.pkl", "wb") as fopen:
            pickle.dump(possible_pairs, fopen)

        return possible_pairs

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
            curr_bias["params"]["use_pbc"] = False # FIXME:
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
        # possible_pairs = self._prepare_fragments(atoms)
        num_atoms = len(atoms)
        fragments = find_molecules(atoms, reactive_indices=range(45, num_atoms))
        self._print(fragments)

        # pair order is random
        possible_pairs = list(itertools.product(fragments["H"], fragments["CO"]))
        for pp in possible_pairs:
            self._print(pp)
        possible_pairs = [([50], [45, 46])]
        self._print(f"{possible_pairs =}")

        num_pairs = len(possible_pairs)

        # serialisation pair data
        pair_data_fpath = self.directory/"pairs.json"
        with open(pair_data_fpath, "w") as fopen:
            json.dump(possible_pairs, fopen)

        # NOTE: Sampled structures are
        #       (num_structures, num_reactions, num_gamma, num_frames)
        #       num_computations = num_structures * num_reactions * num_gamma??

        # TODO: If gamma is too large, the first computation leads to reaction

        comput_dpath = self.directory/"comput"
        comput_dpath.mkdir(exist_ok=True)

        # - run each pair
        grid_workers = []
        for i, rxn_pair in enumerate(possible_pairs):
            # write to log
            self._print(f"===== Pair {i} =====")
            reactants = convert_index_to_formula(atoms, rxn_pair)
            self._print("reactants: ")
            for ir, r in enumerate(reactants):
                self._print(f"  {r:>24s}_{rxn_pair[ir]}")

            # FIXME: batchsize?
            potters, drivers = self._spawn_computers(rxn_pair, self.gamma_factors)
            curr_worker = GridDriverBasedWorker(potters=potters, drivers=drivers)
            curr_worker.directory = comput_dpath / f"pair{i}"
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
                write(ret_dir/f"prp{i}.xyz", pseudo_pathway)

            # dump finished flag
            with open(self.directory/"FINISHED", "w") as fopen:
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

        with open(self.directory/"pairs.json", "r") as fopen:
            pairs = json.load(fopen)
        num_pairs = len(pairs)
        
        workers = []
        for i in range(num_pairs):
            potters, drivers = self._spawn_computers(pairs[i], self.gamma_factors)
            curr_worker = GridDriverBasedWorker(potters=potters, drivers=drivers)
            curr_worker.directory = self.directory/ "comput"/ f"pair{i}"
            assert curr_worker.directory.exists()
            workers.append(curr_worker)

        return workers


if __name__ == "__main__":
    ...
