#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import time
from random import random
import pathlib
from pathlib import Path
import importlib
import warnings
import logging

import numpy as np

import ase.data
import ase.formula

from ase import Atoms
from ase.io import read, write
from ase.ga.data import PrepareDB, DataConnection

from ase.ga.population import Population

from ase.ga.offspring_creator import OperationSelector

"""
TODO: search variational composition

Workflow
    check current calculation
        |
    check population and generate offspring
        |
    submit unrelaxed structures

Systems
    bulk
    slab
    cluster (w/support)

Reserved Keywords in Database
    generation
    relaxed
    queued
    extinct
    description
    pairing

Operators
    comparator
    crossover (pairing)
    mutation
"""

class GeneticAlgorithemEngine():

    """
    Genetic Algorithem Engine
    """

    # output 
    restart = True
    pfunc = print
    _directory = Path.cwd()

    # local optimisation directory
    CALC_DIRNAME = "tmp_folder"
    PREFIX = "cand"

    # reproduction and mutation
    MAX_REPROC_TRY = 1

    # TODO: Neighbor list and parametrization parameters to screen
    # candidates before relaxation can be added. Default is not to use.

    find_neighbors = None
    perform_parametrization = None

    test_dist_to_slab = True
    test_too_far = True

    # - cell settings
    #number_of_variable_cell_vectors=0,
    #box_to_place_in=box_to_place_in,
    #box_volume=None,
    #splits=None,
    #cellbounds=None,


    def __init__(self, ga_dict: dict, directroy=Path.cwd(), *args, **kwargs):
        """"""
        # - 
        self.directory = directroy
        self._init_logger()

        # - 
        self.ga_dict = copy.deepcopy(ga_dict)

        # - check system type
        from GDPy.builder.randomBuilder import RandomGenerator
        self.generator = RandomGenerator(ga_dict["system"])

        # --- database ---
        self.db_name = pathlib.Path(ga_dict["database"])

        # --- calculation ---
        from GDPy.potential.register import create_potter
        self.worker = create_potter(ga_dict["worker"])
        self.worker.directory = Path.cwd() / self.CALC_DIRNAME

        # --- directory ---
        prefix = self.ga_dict.get("prefix", "cand")
        self.PREFIX = prefix

        # --- population ---
        self.population_size = self.ga_dict["population"]["init_size"]
        self.pop_init_seed = self.ga_dict["population"].get("init_seed", None)
        self.pop_tot_size = self.ga_dict["population"].get("tot_size", self.population_size)
        self.pop_ran_size = self.ga_dict["population"].get("ran_size", 0)
        assert self.population_size == self.pop_tot_size, "tot_size should equal pop_size"
        assert self.pop_ran_size < self.population_size, "ran_size should be smaller than pop_size"

        self.pmut = self.ga_dict["population"].get("pmut", 0.5)

        # --- property ---
        self.prop_dict = ga_dict["property"]
        target = self.prop_dict.get("target", "energy")
        self.prop_dict["target"] = target
        self.pfunc(f"\nTarget of Global Optimisation is {target}")

        # --- convergence ---
        self.conv_dict = ga_dict["convergence"]

        return
    
    @property
    def directory(self):
        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        self._directory = Path(directory_)
        return

    def _init_logger(self):
        """"""
        self.logger = logging.getLogger(__name__)

        log_level = logging.INFO

        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        working_directory = self.directory
        log_fpath = working_directory / (self.__class__.__name__+".out")

        if self.restart:
            fh = logging.FileHandler(filename=log_fpath, mode="a")
        else:
            fh = logging.FileHandler(filename=log_fpath, mode="w")

        fh.setLevel(log_level)
        #fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        #ch.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        self.pfunc = self.logger.info

        return
    
    def operate_database(self, removed_ids= None):
        """data"""
        self.da = DataConnection(self.db_name)

        # check queued
        print("before: ")
        for idx, row in enumerate(self.da.c.select("queued=1")):
            key_value_pairs = row["key_value_pairs"]
            content = "id: {}  origin: {}  cand: {}".format(
                row["id"], key_value_pairs["origin"], key_value_pairs["gaid"]
            )
            print(content)
        
        if removed_ids is not None:
            # NOTE: some calculation may be abnormal when creating input files,
            #       so remove queued and in next run it will be created again
            for confid in removed_ids:    
                print("remove ", confid)
                self.da.remove_from_queue(confid)

        print("after: ")
        for idx, row in enumerate(self.da.c.select("queued=1")):
            key_value_pairs = row["key_value_pairs"]
            content = "id: {}  origin: {}  cand: {}".format(
                row["id"], key_value_pairs["origin"], key_value_pairs["gaid"]
            )
            print(content)

        # remove queued
        #for confid in range(11,22):
        #    print('confid ', confid)
        #    da.remove_from_queue(confid)

        # check pairing
        #for idx, row in enumerate(da.c.select('pairing=1')):
        #    print(idx, ' ', row['id'])
        #    #print(row['key_value_pairs'])
        #    print(row['data'])

        return

    def run(self):
        """ main procedure
        """
        # TODO: check database existence and generation number to determine restart
        if not self.db_name.exists():
            self.pfunc("----- create a new database -----")
            self._create_initial_population()
            # make calculation dir
            self.pfunc("----- create a new tmp_folder -----")
            self.__initialise()
        else:
            self.pfunc("restart the database...")
            # balh
            self.__restart()

        # mutation and comparassion operators
        self.pfunc("\n\n===== register operators =====")
        self._register_operators()

        self.pfunc("\n\n===== register population =====")
        # TODO: population settings
        self.form_population()

        # check current generation number
        self.pfunc("\n\n===== Generation Info =====")

        cur_gen = self.da.get_generation_number()
        self.pfunc(f"current generation number: {cur_gen}")
        max_gen = self.conv_dict["generation"]

        # output a few info
        unrelaxed_strus_gen = list(self.da.c.select("relaxed=0,generation=%d" %cur_gen))
        #for row in unrelaxed_strus_gen:
        #    print(row) # NOTE: mark_as_queue unrelaxed_candidate will have relaxed field too...
        unrelaxed_confids = [row["gaid"] for row in unrelaxed_strus_gen]
        num_unrelaxed_gen = len(unrelaxed_confids)

        relaxed_strus_gen = list(self.da.c.select("relaxed=1,generation=%d" %cur_gen))
        relaxed_confids = [row["gaid"] for row in relaxed_strus_gen]
        num_relaxed_gen = len(relaxed_confids)

        self.pfunc(f"number of relaxed in current generation: {num_relaxed_gen}")
        self.pfunc(sorted(relaxed_confids))
        self.pfunc(f"number of unrelaxed in current generation: {num_unrelaxed_gen}")
        self.pfunc(sorted(unrelaxed_confids))

        # --- check generation
        if cur_gen > max_gen and (num_relaxed_gen == num_unrelaxed_gen):
        #if cur_gen > max_gen:
            self.pfunc("reach maximum generation...")
            self.report()
            return
        else:
            self.pfunc("Have not converged...")

        # - run
        # --- initial population
        if cur_gen == 0:
            self.pfunc("\n\n===== Initial Population Calculation =====")
            while (self.da.get_number_of_unrelaxed_candidates()): # NOTE: this uses GADB get_atoms which adds extra_info
                # calculate structures from init population
                atoms = self.da.get_an_unrelaxed_candidate()
                self.pfunc("\n\n ----- start to run structure %s -----" %atoms.info["confid"])
                # NOTE: provide unified interface to mlp and dft
                _ = self.worker.run([atoms]) # retrieve later
                self.da.mark_as_queued(atoms) # this marks relaxation is in the queue

        self.pfunc("\n\n===== Optimisation and Reproduction =====")
        if (
            # nunrelaxed_gen == 0
            #nrelaxed_gen == unrelaxed_gen == self.population_size
            num_unrelaxed_gen < self.population_size
            # TODO: check whether worker can accept new jobs
        ):
            # TODO: can be aggressive, reproduce when relaxed structures are available
            self.pfunc("not enough unrelaxed candidates for generation %d and try to reproduce..." %cur_gen)
            self.pfunc("number before reproduction: {}".format(self.worker.get_number_of_running_jobs() + num_relaxed_gen))
            count, failed = 0, 0
            #if hasattr(self.pairing, "minfrac"):
            #    previous_minfrac = self.pairing.minfrac
            while (
                self.worker.get_number_of_running_jobs() + num_relaxed_gen < self.population_size
            ):
                #if failed >= 10:
                #    if hasattr(self.pairing, "minfrac"):
                #        self.pairing.minfrac = 0
                #        print("switch minfrac to zero...")
                #    else:
                #        print("too many failures...")
                #        break
                self.pfunc(f"\n\n ----- try to reproduce, count {count}, failed {failed} -----")
                atoms = self._reproduce()
                if atoms:
                    # run opt
                    self.pfunc("\n\n ----- start to run structure %s -----" %atoms.info["confid"])
                    _ = self.worker.run([atoms]) # retrieve later
                    self.da.mark_as_queued(atoms) # this marks relaxation is in the queue
                    count += 1
                else:
                    failed += 1
            else:
                self.pfunc(f"{count} candidates were reproduced in this run...")
                self.pfunc("enough jobs are running for the current generation...")
            #if hasattr(self.pairing, "minfrac"):
            #    self.pairing.minfrac = previous_minfrac 
            #    self.pfunc(f"switch minfrac to {self.pairing.minfrac}...")
        else:
            self.pfunc("Enough candidates or not finished relaxing current generation...")

        # --- check if there were finished jobs
        self.worker.inspect()
        converged_candidates = self.worker.retrieve()
        for cand in converged_candidates:
            # TODO: use tags
            # update extra info
            extra_info = dict(
                data = {},
                key_value_pairs = {"extinct": 0}
            )
            cand.info.update(extra_info)
            # get tags
            confid = cand.info["confid"]
            if self.use_tags:
                rows = list(self.da.c.select(f"relaxed=0,gaid={confid}"))
                for row in rows:
                    if row.formula:
                        previous_atoms = row.toatoms(add_additional_information=True)
                        previous_tags = previous_atoms.get_tags()
                        self.pfunc(f"tags: {previous_tags}")
                        break
                else:
                    raise RuntimeError(f"Cant find tags for cand {confid}")
                cand.set_tags(previous_tags)
            # evaluate raw score
            self.evaluate_candidate(cand)
            self.pfunc(f"  add relaxed cand {confid}")
            self.pfunc("  with raw_score {:.4f}".format(cand.info["key_value_pairs"]["raw_score"]))
            self.da.add_relaxed_step(
                cand,
                find_neighbors=self.find_neighbors,
                perform_parametrization=self.perform_parametrization
            )

        return
    
    def __initialise(self):
        # get basic system information
        self.atom_numbers_to_optimize = self.da.get_atom_numbers_to_optimize()
        self.n_to_optimize = len(self.atom_numbers_to_optimize)
        self.slab = self.da.get_slab()

        self.use_tags = self.generator.use_tags
        self.blmin = self.generator.blmin

        return

    def __restart(self):
        """"""
        # basic system info
        self.da = DataConnection(self.db_name)

        # get basic system information
        self.__initialise()

        # set bond list minimum
        self.generator._print_blmin(self.blmin)

        return
    
    def report(self):
        self.pfunc('restart the database...')
        self.__restart()
        results = pathlib.Path.cwd() / 'results'
        if not results.exists():
            results.mkdir()

        # - write structures
        all_relaxed_candidates = self.da.get_all_relaxed_candidates()
        write(results / 'all_candidates.xyz', all_relaxed_candidates)

        # - plot population evolution
        data = []
        cur_gen_num = self.da.get_generation_number() # equals finished generation plus one
        self.pfunc(f"Current generation number: {cur_gen_num}")
        for i in range(cur_gen_num):
            #print('generation ', i)
            energies = [
                atoms.get_potential_energy() for atoms in all_relaxed_candidates 
                    if atoms.info['key_value_pairs']['generation']==i
            ]
            self.pfunc(energies)
            data.append([i, energies])
        
        import matplotlib as mpl
        mpl.use("Agg") #silent mode
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
        ax.set_title(
            "Population Evolution", 
            fontsize=20, 
            fontweight='bold'
        )

        for i, energies in data:
            ax.scatter([i]*len(energies), energies)
        
        plt.savefig(results/"pop.png")

        return

    def refine(
        self, 
        number=50, # minimum number of structures selected
        aediff=0.05 # maximum atomic energy difference to the putative global minimum
    ):
        """ refine structures with DFT (VASP)
            the number of structures is determined by the rule
        """
        self.pfunc('restart the database...')
        self.__restart()
        results = pathlib.Path.cwd() / "results"
        if not results.exists():
            results.mkdir()
        all_relaxed_candidates = self.da.get_all_relaxed_candidates()
        sorted_candidates = sorted(
            all_relaxed_candidates, key=lambda atoms:atoms.info['key_value_pairs']['raw_score'],
            reverse=True
        )
        nframes = len(sorted_candidates)
        energies = np.array([a.get_potential_energy() for a in sorted_candidates])
        natoms_array = np.array([len(a) for a in sorted_candidates]) # TODO: change this to the number of explored atoms
        atomic_energies = energies / natoms_array
        min_ae = atomic_energies[0] # minimum atomic energy

        for i in range(len(atomic_energies)):
            if atomic_energies[i] >= min_ae + aediff:
                new_number = i
                self.pfunc(f"There are {new_number} structures in the range.")
                break
        else:
            self.pfunc("All structures are in the energy range.")
        number = np.min([number, new_number])

        self.pfunc(f"Select {number} structures out of {nframes}...")
        mosted = sorted_candidates[:number]
        #for atoms in mosted:
        #    self.pfunc(atoms.info['confid'], 'raw_score: ', atoms.info['key_value_pairs']['raw_score'])
        self.pfunc(f"energy range: {mosted[0].get_potential_energy()}  {mosted[-1].get_potential_energy()}")
        saved_xyz = results / (Path.cwd().name + f"-accurate-{number}.xyz")
        write(saved_xyz, mosted)

        """
        from GDPy.ga.make_all_vasp import create_by_ase
        for atoms in mosted:
            dname = pathlib.Path.cwd() / 'accurate' / ('cand{0}'.format(atoms.info['confid']))
            create_by_ase(
                atoms, self.ga_dict["postprocess"]["incar"],
                dname
            )
        """
        self.pfunc("create refinement directory...")
        from GDPy.utils.data import vasp_creator, vasp_collector
        incar_template = self.ga_dict["postprocess"]["incar"]
        # prefix = Path.cwd() / "accurate"
        prefix = Path("/mnt/scratch2/users/40247882/oxides/eann-main/train-all/m25r/ga-Pt322-fp")
        if not prefix.exists():
            prefix.mkdir()
        else:
            self.pfunc("skip accurate...")

        vasp_creator.create_files(
            prefix,
            "/users/40247882/repository/GDPy/GDPy/utils/data/vasp_calculator.py",
            incar_template,
            saved_xyz,
            #to_submit = False
            to_submit = True
        )

        return
    
    def __get_operator(self, operators, settings, default_op):
        """ comparator, crossover, mutation
        """
        #self.pfunc("operators: ", operators)
        #comp_settings = op_dict.get(section, None)
        if settings is not None:
            # - get operator
            op_name = settings.get("name", None)
            assert op_name is not None, f"No op name is provided."
            kwargs = settings.get("kwargs", None)
        else:
            op_name = default_op
            kwargs = None

        #self.pfunc(f"use {op_name}")
        op_obj = getattr(operators, op_name)
        default_params = getattr(operators, op_obj.__name__+"_params")

        return op_obj, default_params, kwargs
    
    def _register_operators(self):
        """ register various operators
            comparator, pairing, mutations
        """
        op_dict = self.ga_dict.get("operators", None)
        if op_dict is None:
            op_dict = {
                "comparator": {"name": "InteratomicDistanceComparator"},
                "crossover": {"name": "CutAndSplicePairing"}
            }
        operators = importlib.import_module("GDPy.ga.operators")

        # --- comparator
        comparator, params, kwargs = self.__get_operator(
            operators, op_dict.get("comparator", None), 
            "InteratomicDistanceComparator"
        )
        # - update params based on this system
        if "n_top" in params.keys():
            params["n_top"] = self.n_to_optimize
        if isinstance(kwargs, dict):
            params.update(**kwargs)
        self.comparing = comparator(**params)

        self.pfunc("--- comparator ---")
        self.pfunc(f"Use comparator {comparator.__name__}.")

        # --- crossover
        #self.pfunc("operators: ", operators)
        crossover, params, kwargs = self.__get_operator(
            operators, op_dict.get("crossover", None), 
            "CutAndSplicePairing"
        )
        if "slab" in params.keys():
            params["slab"] = self.slab
        if "n_top" in params.keys():
            params["n_top"] = self.n_to_optimize
        if "blmin" in params.keys():
            params["blmin"] = self.blmin
        if "use_tags" in params.keys():
            params["use_tags"] = self.use_tags
        if isinstance(kwargs, dict):
            params.update(**kwargs)
        #self.pfunc("pairing params: ")
        #for k, v in params.items():
        #    self.pfunc(k, "->", v)
        self.pairing = crossover(**params)
        #self.pairing = CutAndSplicePairing(self.slab, self.n_to_optimize, self.blmin)

        self.pfunc("--- crossover ---")
        self.pfunc(f"Use crossover {crossover.__name__}.")
        #self.pfunc("pairing: ", self.pairing)

        # --- mutations
        mutations, probs = [], []
        mutation_list = op_dict.get("mutation", None)
        #self.pfunc(mutation_list)
        if isinstance(mutation_list, list):
            for mut_settings in mutation_list:
                mut, params, kwargs = self.__get_operator(
                    operators, mut_settings, "RattleMutation"
                )
                if "n_top" in params.keys():
                    params["n_top"] = self.n_to_optimize
                if "blmin" in params.keys():
                    params["blmin"] = self.blmin
                if "use_tags" in params.keys():
                    params["use_tags"] = self.use_tags
                # NOTE: check this mutation whether valid for this system
                if kwargs is None:
                    prob = 1.0
                else:
                    prob = kwargs.pop("prob", 1.0)
                probs.append(prob)
                if isinstance(kwargs, dict):
                    params.update(**kwargs)
                mutations.append(mut(**params))
        else:
            # NOTE: default only has one RattleMutation
            mut, params, kwargs = self.__get_operator(
                operators, None, "RattleMutation"
            )
            if "n_top" in params.keys():
                params["n_top"] = self.n_to_optimize
            if "blmin" in params.keys():
                params["blmin"] = self.blmin
            if "use_tags" in params.keys():
                params["use_tags"] = self.use_tags
            if kwargs is None:
                prob = 1.0
            else:
                prob = kwargs.pop("prob", 1.0)
            probs.append(prob)
            if isinstance(kwargs, dict):
                params.update(**kwargs)
            mutations.append(mut(**params))

        self.pfunc("--- mutations ---")
        self.pfunc(f"mutation probability: {self.pmut}")
        for mut in mutations:
            self.pfunc(f"Use mutation {mut.descriptor}.")
        self.mutations = OperationSelector(probs, mutations, rng=np.random)

        return
    
    def _create_initial_population(
            self, 
        ):
        # create the database to store information in
        # TODO: move this part to where before generator is created
        da = PrepareDB(
            db_file_name = self.db_name,
            simulation_cell = self.generator.slab,
            stoichiometry = self.generator.atom_numbers_to_optimise
        )

        self.pfunc("\n\n===== Initial Population Creation =====")
        # read seed structures
        if self.pop_init_seed is not None:
            self.pfunc("----- try to add seed structures -----")
            seed_frames = read(self.pop_init_seed, ":")
            seed_size = len(seed_frames)
            assert (seed_size > 0 and seed_size <= self.population_size), "number of seeds is invalid"
            # NOTE: check force convergence and only add converged structures
            # check atom permutation
            for i, atoms in enumerate(seed_frames):
                # TODO: check atom order
                atoms.info["data"] = {}
                atoms.info["key_value_pairs"] = {}
                atoms.info["key_value_pairs"]["origin"] = "seed {}".format(i)
                atoms.info["key_value_pairs"]["raw_score"] = -atoms.get_potential_energy()
                # TODO: check geometric convergence
                if True: # force converged
                    self.pfunc(f"  add converged seed {i}")
                    da.add_relaxed_candidate(atoms)
                else:
                    # run opt
                    pass
        else:
            seed_size = 0

        # generate the starting population
        self.pfunc("start to create initial population")
        starting_population = self.generator.run(self.population_size - seed_size)
        #self.pfunc("start: ", starting_population)
        self.pfunc(f"finished creating initial population...")

        self.pfunc(f"save population {len(starting_population)} to database")
        for a in starting_population:
            da.add_unrelaxed_candidate(a)
        
        # TODO: change this to the DB interface
        self.pfunc("save population size {0} into database...".format(self.population_size))
        row = da.c.get(1)
        new_data = row["data"].copy()
        new_data['population_size'] = self.population_size
        da.c.update(1, data=new_data)

        self.da = DataConnection(self.db_name)

        return
    
    def add_seed_structures(self, spath):
        """ add structures into database
            can be done during any time in global optimisation
        """

        return

    def form_population(self):
        """"""
        # set current population
        # usually, it should be the same as the initial size
        # but for variat composition search, a large init size can be useful

        # create the population
        self.population = Population(
            data_connection = self.da,
            population_size = self.population_size,
            comparator = self.comparing
        )

        # self.pfunc out population info
        #frames = self.population.get_current_population()
        #self.pfunc('current population size: ', len(frames))
        #for atoms in frames:
        #    n_paired = atoms.info.get('n_paired', None)
        #    looks_like = atoms.info.get('looks_like', None)
        #    self.pfunc(atoms.info['confid'], ' -> ', n_paired, ' -> ', looks_like)

        return
    
    def _reproduce(self):
        """generate an offspring"""
        # Submit new candidates until enough are running
        a3 = None
        for i in range(self.MAX_REPROC_TRY):
            # try 10 times
            parents = self.population.get_two_candidates()
            a3, desc = self.pairing.get_new_individual(parents) # NOTE: this also adds key_value_pairs to a.info
            if a3 is not None:
                self.da.add_unrelaxed_candidate(
                    a3, description=desc # here, desc is used to add "pairing": 1 to database
                ) # if mutation happens, it will not be relaxed

                mut_desc = ""
                if random() < self.pmut:
                    a3_mut, mut_desc = self.mutations.get_new_individual([a3])
                    #self.pfunc("a3_mut: ", a3_mut.info)
                    #self.pfunc("mut_desc: ", mut_desc)
                    if a3_mut is not None:
                        self.da.add_unrelaxed_step(a3_mut, mut_desc)
                        a3 = a3_mut
                self.pfunc(f"  generate offspring with {desc} \n  {mut_desc} after {i+1} attempts..." )
                break
            else:
                mut_desc = ""
                self.pfunc(f"  failed generating offspring with {desc} \n  {mut_desc} after {i+1} attempts..." )
        else:
            self.pfunc("cannot generate offspring a3 after {0} attempts".format(self.MAX_REPROC_TRY))

        return a3
    
    def evaluate_candidate(self, atoms):
        """ TODO: evaluate candidate based on raw score
            in most cases, it's potential energy
            but this is should be more flexible
            e.g. enthalpy (pot+pressure), reaction energy
        """
        assert atoms.info["key_value_pairs"].get("raw_score", None) is None, "candidate already has raw_score before evaluation"
        
        # NOTE: larger raw_score, better candidate

        # evaluate based on target property
        target = self.prop_dict["target"]
        if target == "energy":
            atoms.info["key_value_pairs"]["raw_score"] = -atoms.get_potential_energy()
        elif target == "barrier":
            pass
        else:
            raise RuntimeError(f"Unknown target {target}...")

        return

if __name__ == "__main__":
    pass
