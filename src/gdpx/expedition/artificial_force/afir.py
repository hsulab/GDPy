#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import logging
import pathlib
import time
import pickle

from typing import NoReturn, List, Mapping

from itertools import combinations, product, chain, groupby

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.formula import Formula

from .. import registers
from .. import StructureBuilder
from .. import ComputerVariable, DriverBasedWorker
from ..expedition import AbstractExpedition

from gdpx.worker.grid import GridDriverBasedWorker
from gdpx.potential.interface import create_mixer
from gdpx.builder.group import create_a_group, create_a_molecule_group
from gdpx.graph.molecule import find_product, find_molecules


def convert_index_to_formula(atoms, group_indices: List[List[int]]):
    """"""
    formulae = []
    for g in group_indices:
        symbols = [atoms[i].symbol for i in g]
        formulae.append(
            Formula.from_list(symbols).format("hill")
        )
    #formulae = sorted(formulae)

    return formulae


def find_target_fragments(atoms, target_commands: List[str]) -> Mapping[str,List[List[int]]]:
    """Find target fragments in the structure to react.

    This is a wrapper for group commands as there are several ways to defind
    a (atomic) group but we need a group of molecules here. Optional ways are 
    using `molecule` all, using `tag`...

    """
    fragments = {} # Mapping[str,List[List[int]]]
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


class AFIRSearch(AbstractExpedition):


    def __init__(
            self, builder, target, gamma: List[float]=[0.5, 2.5, 1.0], 
            seed=None, directory="./", *args, **kwargs
        ) -> None:
        """Define some basic parameters for the afir search.

        Args:
            builder: StructureBuilder.
        
        """
        self.directory = directory

        self.builder = builder
        self.target = target
        #assert len(self.target) == 2, "Target only supports two elements."
        self.gamma = gamma

        self.bias_params = dict(
            name = "bias", 
            params = dict(
                backend="ase", type = "afir",
                gamma = None, groups = None
            )
        )

        return

    def register_worker(self, worker: dict, *args, **kwargs):
        """"""
        if isinstance(worker, dict):
            worker_params = copy.deepcopy(worker)
            worker = registers.create(
                "variable", "computer", convert_name=True, **worker_params
            ).value[0]
        elif isinstance(worker, list): # assume it is from a computervariable
            worker = worker[0]
        elif isinstance(worker, ComputerVariable):
            worker = worker.value[0]
        elif isinstance(worker, DriverBasedWorker):
            worker = worker
        else:
            raise RuntimeError(f"Unknown worker type {worker}")
        
        self.worker = worker

        return

    def _prepare_fragments(self, atoms, *args, **kwargs):
        """"""
        data_path = self.directory / "_data"
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)

        # - find possible reaction pairs
        frag_fpath = data_path/"fragments.pkl"
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
        
        with open(self.directory/"_data"/"pairs.pkl", "wb") as fopen:
            pickle.dump(possible_pairs, fopen)

        return possible_pairs
    
    def _create_afir_potters(self, pair: List[List[int]], gamma_list: List[float], *args, **kwargs):
        """"""
        if hasattr(self.worker.potter, "remove_loaded_models"):
            self.worker.potter.remove_loaded_models()

        bias_list = []
        for g in gamma_list:
            curr_bias = copy.deepcopy(self.bias_params)
            curr_bias["params"]["groups"] = pair 
            curr_bias["params"]["gamma"] = g
            bias_list.append(curr_bias)
        potters = [create_mixer(self.worker.potter, b) for b in bias_list]

        return potters
    
    def run(self, *args, **kwargs) -> None:
        """"""
        # - some imported packages change `logging.basicConfig` 
        #   and accidently add a StreamHandler to logging.root
        #   so remove it...
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logging.root.removeHandler(h)

        self._print("---------------------------")
        self._print("| AFIRSearch starts...... |")
        self._print("---------------------------")
        # TODO: check restart

        # - assume input structures are minimised
        frames = self.builder.run()
        nframes = len(frames)
        assert nframes == 1, "Only one structure for now."
        atoms = frames[0]

        # - find fragments
        possible_pairs = self._prepare_fragments(atoms)
        
        # - prepare afir bias
        gamma_list = np.linspace(*self.gamma, endpoint=True)

        # - run each pair
        grid_workers = []
        for i, pair in enumerate(possible_pairs):
            # -- start info
            self._print(f"===== Pair {i} =====")
            reactants = convert_index_to_formula(atoms, pair)
            self._print("Reactants:")
            self._print(reactants)
            self._print(pair)

            # -- create GridWorker
            potters = self._create_afir_potters(pair, gamma_list)
            curr_worker = GridDriverBasedWorker(
                potters, driver=self.worker.driver
            )
            curr_worker.directory = self.directory / f"pair{i}"
            grid_workers.append(curr_worker)
            curr_worker.run([atoms])
        
        # - extract each pair
        worker_status = []
        for i, curr_worker in enumerate(grid_workers):
            curr_worker.inspect(resubmit=True)
            if curr_worker.get_number_of_running_jobs() == 0:
                worker_status.append(True)
            else:
                worker_status.append(False)
        
        if all(worker_status):
            self._print("---------------------------")
            self._print("| AFIRSearch is finished. |")
            self._print("---------------------------")
            nimages = len(gamma_list)
            for i, (pair, curr_worker) in enumerate(zip(possible_pairs, grid_workers)):
                self._print(f"===== Pair {i} =====")
                reactants = convert_index_to_formula(atoms, pair)
                self._print("Reactants:")
                self._print(reactants)
                self._print(pair)
                curr_trajs = curr_worker.retrieve(include_retrieved=True)
                pseudo_pathway = [t[-1] for t in curr_trajs]
                end_atoms = pseudo_pathway[-1]
                prod_indices = find_product(end_atoms, pair)
                products = convert_index_to_formula(end_atoms, prod_indices)
                self._print("Products:")
                self._print(products)
                if sorted(reactants) == sorted(products):
                    # TODO: index may change? molecule reconstruct?
                    self._print("nothing happens...")
                else:
                    self._print("reaction happens...")
                    # TODO: postprocess, minimise FS to create a pathway
            # - 
            with open(self.directory/"FINISHED", "w") as fopen:
                fopen.write(
                    f"FINISHED AT {time.asctime( time.localtime(time.time()) )}."
                )
        else:
            self._print("Some calculations are unfinished.")

        return
    
    def read_convergence(self, *args, **kwargs):
        """"""
        converged = False
        if (self.directory/"FINISHED").exists():
            converged = True

        return converged
    
    def get_workers(self, *args, **kwargs):
        """"""
        workers = []
        if hasattr(self.worker.potter, "remove_loaded_models"):
            self.worker.potter.remove_loaded_models()

        pair_data_path = self.directory/"_data"/"pairs.pkl"
        if pair_data_path.exists():
            with open(pair_data_path, "rb") as fopen:
                pairs = pickle.load(fopen)
            wdirs = sorted(
                list(self.directory.glob("pair*")), key=lambda x: int(x.name[4:])
            )
            assert len(pairs) == len(wdirs), f"Inconsistent number of pairs {len(pairs)} and wdirs {len(wdirs)}."

            gamma_list = np.linspace(*self.gamma, endpoint=True)
            for i, pair in enumerate(pairs):
                potters = self._create_afir_potters(pair, gamma_list)
                curr_worker = GridDriverBasedWorker(
                    potters, driver=self.worker.driver
                )
                curr_worker.directory = self.directory / f"pair{i}"
                workers.append(curr_worker)
        else:
            raise FileNotFoundError("There is no cache for pairs.")
        
        return workers


if __name__ == "__main__":
    ...