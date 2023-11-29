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

from .. import Variable
from .. import registers
from .. import StructureBuilder
from .. import ComputerVariable, DriverBasedWorker
from ..expedition import AbstractExpedition

from GDPy.potential.interface import create_mixer
from GDPy.builder.group import create_a_group, create_a_molecule_group
from GDPy.computation.utils import make_clean_atoms
from GDPy.graph.creator import find_product, find_molecules


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


class ArtificialReactionVariable(Variable):

    def __init__(self, builder, directory="./", *args, **kwargs) -> None:
        """"""
        if isinstance(builder, dict):
            builder_params = copy.deepcopy(builder)
            builder_method = builder_params.pop("method")
            builder = registers.create(
                "builder", builder_method, convert_name=False, **builder_params
            )
        elif isinstance(builder, StructureBuilder):
            builder = builder
        else:
            raise RuntimeError(f"Unknown type {type(StructureBuilder)} for Builder.")

        engine = AFIRSearch(builder, *args, **kwargs)

        super().__init__(initial_value=engine, directory=directory)

        return


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


PATHWAY_FNAME = "pseudo_path.xyz"

class AFIRSearch(AbstractExpedition):

    is_restart = False

    def __init__(
            self, builder, target, mechanism: str="bi", gamma: List[float]=[0.5,2.5,1.0], 
            min_is=True, find_mep=False,
            seed=None, directory="./", *args, **kwargs
        ) -> None:
        """Define some basic parameters for the afir search.

        Args:
            builder: StructureBuilder.
            min_is: Whether to minimise the initial structure.
                    Otherwise, the single-point energy is calculated.
        
        """
        self.builder = builder

        self.target = target
        #assert len(self.target) == 2, "Target only supports two elements."

        self.mechanism = mechanism # default should be bimolecular reaction

        self.gamma = gamma
        
        self.min_is = min_is
        self.find_mep = find_mep

        return
    
    def _restart(self):
        """Restart search.

        TODO: Check whether parameters (afir, atoms, worker) has been changed.
        
        """


        return

    def _prepare_fragments(self, ):
        """"""

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
    
    def run(self, *args, **kwargs) -> None:
        """"""
        # - some imported packages change `logging.basicConfig` 
        #   and accidently add a StreamHandler to logging.root
        #   so remove it...
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logging.root.removeHandler(h)

        self._print("START AFIR SEARCH")
        # TODO: check restart

        # - assume input structures are minimised
        frames = self.builder.run()
        self._print(frames)
        nframes = len(frames)
        assert nframes == 1, "Only one structure for now."
        atoms = frames[0]

        # - find possible reaction pairs
        frag_fpath = self.directory/"fragments.pkl"
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

        rxn_fpath = self.directory/"rxn.dat"
        if not rxn_fpath.exists():
            with open(rxn_fpath, "w") as fopen:
                fopen.write(
                    ("{:<8s}  "*2+"{:<12s}  "*6+"{:<12s}  "+"\n").format(
                        "#Pair", "nframes", "ene_is", "frc_is",
                        "ene_ts", "frc_ts", "ene_fs", "frc_fs",
                        "is_reacted"
                    )
                )
        else:
            rxn_data = []
            with open(rxn_fpath, "r") as fopen:
                lines = fopen.readlines()
                for line in lines:
                    rxn_data.append(line.strip().split())
            nfinished = len(rxn_data) - 1
            self._print("".join(lines))
            self._print(f"finished pairs: {nfinished}")
            possible_pairs = possible_pairs[nfinished:]
        
        # - prepare afir bias
        # TODO: retain bias in the driver?
        bias_params = dict(
            name = "bias", 
            params = dict(
                backend="ase", type = "afir",
                gamma = None, groups = None
            )
        )
        self._print(self.worker)

        # - run each pair
        for i, pair in enumerate(possible_pairs):
            # -- start info
            self._print(f"===== Pair {i} =====")
            bias_params["params"]["groups"] = pair 
            reactants = convert_index_to_formula(atoms, pair)
            self._print("Reactants:")
            self._print(reactants)
            self._print(pair)

            # -- run pair
            directory_ = self.directory / f"p{i}"
            self._irun(atoms, self.worker, reactants, bias_params, directory_)

        return
    
    def _irun(self, atoms_: Atoms, worker, reactants: List[str], bias_params_: dict, directory_):
        """Run a specific reaction pair in the structure."""
        ginit, gmax, gintv = self.gamma # eV
        # TODO: if ginit is none, then random one
        if ginit is None:
            cur_gamma = gmax*self.rng.random()
        else:
            cur_gamma = ginit
        self._print(f"Initial Gamma: {cur_gamma}")

        driver = worker.driver

        reactants = sorted(reactants)

        # - run biased minimisation
        path_frames = [atoms_] # add initial state
        is_reacted = False

        ngamma = 0
        while cur_gamma <= gmax:
            self._print(f"Current Gamma: {cur_gamma}")
            # TODO: make this part parallel, which maybe efficient for
            #       ab initio calculations...
            cur_atoms = atoms_.copy()
            # -- update worker's bias
            bias_params = copy.deepcopy(bias_params_)
            bias_params["params"]["gamma"] = cur_gamma
            self._print(bias_params)

            # -- create mixer
            mixer = create_mixer(worker.potter.as_dict(), bias_params)
            self._print(f"Mixer: {mixer}")
            self._print(f"Driver: {driver.as_dict()}")
            #driver.bias = driver._parse_bias([bias_params])
            #print(driver.bias)
            #print(driver.bias[0].gamma)
            driver = mixer.create_driver(driver.as_dict())
            self._print(f"driver: {driver}")

            # -- run calculation
            driver.directory = directory_ / f"g{ngamma}"
            end_atoms_ = driver.run(cur_atoms) # end frame of the minimisation

            results = dict(
                energy = end_atoms_.get_potential_energy(),
                forces = end_atoms_.get_forces(apply_constraint=False)
            )
            end_atoms = make_clean_atoms(end_atoms_, results)

            path_frames.append(end_atoms)

            # -- check if reaction happens?
            #    if so, break while
            prod_indices = find_product(end_atoms, bias_params["groups"])
            products = convert_index_to_formula(end_atoms, prod_indices)
            products = sorted(products)
            self._print("Molecules: {}".format(" ".join(products)))
            if products == reactants:
                # TODO: index may change? molecule reconstruct?
                self._print("nothing happens...")
                ...
            else:
                # TODO: resacale gamma if reaction were found in the second minimisation
                is_reacted = True
                self._print("reaction happens...")
                self._print(
                    "{} -> {}".format(" + ".join(reactants), " + ".join(products))
                )
                self._print("minimise the last structure to the final state...")
                fs_atoms = self._minimise_stable_state(end_atoms, worker, directory_=directory_/"fs")
                path_frames.append(fs_atoms)
                break

            # -- update gamma
            cur_gamma += gintv
            ngamma += 1
        
        # - sort results
        write(directory_/PATHWAY_FNAME, path_frames)

        nframes = len(path_frames)
        energies = [a.get_potential_energy() for a in path_frames]
        ene_ts = max(energies)
        ts_idx = energies.index(ene_ts)

        maxforces = [np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in path_frames]
        frc_ts = maxforces[ts_idx]

        # info: nframes, IS, TS, FS, is_reacted?
        with open(self.directory/"rxn.dat", "a") as fopen:
            fopen.write(
                ("{:<8s}  "+"{:<8d}  "+"{:<12.4f}  "*6+"{:<12s}"+"\n").format(
                    directory_.name, nframes, energies[0], maxforces[0],
                    ene_ts, frc_ts, energies[-1], maxforces[-1], str(is_reacted)
                )
            )

        return
    
    def _minimise_stable_state(self, atoms_, worker, directory_, **kwargs) -> Atoms:
        """Re-minimise stable states (IS and FS).

        Initial state (input structure) may be minimised at an another theory 
        level or even not minimised at all.

        """
        # - prepare driver, remove bias
        driver = worker.driver
        driver._parse_bias([])
        driver.directory = directory_

        cur_atoms = atoms_.copy()
        end_atoms_ = driver.run(cur_atoms, **kwargs) # end frame of the minimisation

        results = dict(
            energy = end_atoms_.get_potential_energy(),
            forces = end_atoms_.get_forces(apply_constraint=False)
        )
        end_atoms = make_clean_atoms(end_atoms_, results)

        return end_atoms
    
    def report(self, worker: DriverBasedWorker, *args, **kwargs) -> Mapping[str,List[Atoms]]:
        """Report results."""
        # NOTE: convention cand0 - pair0 - gamma0 - step0
        # - need driver to read trajectories
        driver = worker.driver

        # - read rxn data
        rxn_data = []
        with open(self.directory/"rxn.dat", "r") as fopen:
            lines = fopen.readlines()[1:]
            for line in lines:
                rxn_data.append(line.strip().split())
        pairs = [x[0] for x in rxn_data] # TODO: only collect is_reacted?

        # - run over pairs
        ret = dict(
            pathways = [], # List[List[Atoms]] NOTE: IS are all the same...
            trajs = [] # List[List[List[Atoms]]] NOTE: first structures are all the same...
        )
        #print(pairs)
        for p in pairs:
            pair_wdir = self.directory/p
            # -- find gammas and fs
            path_frames = read(pair_wdir/PATHWAY_FNAME, ":")
            ret["pathways"].append(path_frames) # add results
            nframes_path = len(path_frames)
            if (pair_wdir/"fs").exists():
                # both have is and fs
                gamma_wdirs = [f"g{i}" for i in range(nframes_path-2)]
                gamma_wdirs.append("fs")
            else:
                # only have is
                gamma_wdirs = [f"g{i}" for i in range(nframes_path-1)]
            #print(pair_wdir, gamma_wdirs)
            gamma_trajectories = []
            for g in gamma_wdirs:
                #print(g)
                driver.directory = pair_wdir/g
                traj_frames = driver.read_trajectory()
                gamma_trajectories.append(traj_frames)
                #print(traj_frames)
            ret["trajs"].append(gamma_trajectories)
        
        # - results

        return ret
    
    def read_convergence(self):
        return super().read_convergence()
    
    def get_workers(self):
        return super().get_workers()


if __name__ == "__main__":
    ...