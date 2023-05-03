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

from GDPy.ga.population import AbstractPopulationManager

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

        # --- population ---
        self.pop_manager = AbstractPopulationManager(ga_dict["population"])
        self.pop_manager.pfunc = self.logger.info

        # --- property ---
        self.prop_dict = ga_dict["property"]
        target = self.prop_dict.get("target", "energy")
        self.prop_dict["target"] = target

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
    
    def run(self, worker, steps=None):
        """Run the GA procedure several steps.

        Default setting would run the algorithm many times until its convergence. 
        This is useful for running optimisations with serial worker.

        """
        conv_gen_num = self.conv_dict["generation"]

        if steps is None:
            steps = conv_gen_num

        for istep in range(steps):
            self._irun(worker, istep)

        return

    def _irun(self, worker, istep: int):
        """ main procedure
        """
        # - search target
        self.pfunc(f"\n\n===== Genetic Algorithm at Step {istep} =====")
        target = self.prop_dict["target"]
        self.pfunc(f"\nTarget of Global Optimisation is {target}")

        # - worker info
        self.pfunc("\n\n===== register worker =====")
        assert worker is not None, "Worker is not properly set..."
        self.worker = worker
        self.worker.directory = self.directory / self.CALC_DIRNAME
        self.worker.logger = self.logger

        # - generator info
        self.pfunc("\n\n===== register generator =====")
        self.pfunc(self.generator)

        # TODO: check database existence and generation number to determine restart
        if not self.db_name.exists():
            self.pfunc("----- create a new database -----")
            self._create_initial_population()
            # make calculation dir
            self.pfunc("----- create a new tmp_folder -----")
            self.__initialise()
        else:
            self.pfunc("restart the database...")
            # blah
            self.__restart()

        # --- mutation and comparassion operators
        self.pfunc("\n\n===== register operators =====")
        self._register_operators()

        # --- check current generation number
        self.pfunc("\n\n===== Generation Info =====")

        self.cur_gen = self.da.get_generation_number()
        self.pfunc(f"current generation number: {self.cur_gen}")

        # output a few info
        unrelaxed_strus_gen_ = list(self.da.c.select("relaxed=0,generation=%d" %self.cur_gen))
        unrelaxed_strus_gen = []
        for row in unrelaxed_strus_gen_:
            # NOTE: mark_as_queue unrelaxed_candidate will have relaxed field too...
            if "queued" not in row:
                unrelaxed_strus_gen.append(row)
        unrelaxed_confids = [row["gaid"] for row in unrelaxed_strus_gen]
        self.num_unrelaxed_gen = len(unrelaxed_confids)

        relaxed_strus_gen = list(self.da.c.select("relaxed=1,generation=%d" %self.cur_gen))
        for row in relaxed_strus_gen:
            print(row)
        relaxed_confids = [row["gaid"] for row in relaxed_strus_gen]
        self.num_relaxed_gen = len(relaxed_confids)

        # check if this is the end of the current generation
        end_of_gen = (self.num_relaxed_gen == self.num_unrelaxed_gen)

        self.pfunc(f"number of relaxed in current generation: {self.num_relaxed_gen}")
        self.pfunc(sorted(relaxed_confids))
        self.pfunc(f"number of unrelaxed in current generation: {self.num_unrelaxed_gen}")
        self.pfunc(sorted(unrelaxed_confids))
        self.pfunc(f"end of current generation: {end_of_gen}")

        # --- check generation
        if self.is_converged():
            self.pfunc("reach maximum generation...")
            return
        else:
            self.pfunc("not converged yet...")

        # - run
        self.pfunc("\n\n===== Population Info =====")
        #content = "For generation == 0 (initial population),"
        #content += "{:>8s}  {:>8s}  {:>8s}\n".format("Random", "Seed", "Total")
        #content += "{:>8d}  {:>8d}  {:>8d}\n".format(
        #    self.pop_manager.gen_rep_size, self.pop_manager.gen_ran_size,
        #)
        content = "For generation > 0,\n"
        content += "{:>8s}  {:>8s}  {:>8s}  {:>8s}\n".format(
            "Reprod", "Random", "Mutate", "Total"
        )
        content += "{:>8d}  {:>8d}  {:>8d}  {:>8d}\n".format(
            self.pop_manager.gen_rep_size, self.pop_manager.gen_ran_size,
            self.pop_manager.gen_mut_size, self.pop_manager.gen_size
        )
        content += "Note: Reproduced structure has a chance (pmut) to mutate.\n"
        self.pfunc(content)

        # --- initial population
        if self.cur_gen == 0:
            self.pfunc("\n\n===== Initial Population Calculation =====")
            frames_to_work = []
            while (self.da.get_number_of_unrelaxed_candidates()): # NOTE: this uses GADB get_atoms which adds extra_info
                # calculate structures from init population
                atoms = self.da.get_an_unrelaxed_candidate()
                self.pfunc("\n\n ----- start to run structure %s -----" %atoms.info["confid"])
                frames_to_work.append(atoms)
                self.da.mark_as_queued(atoms) # this marks relaxation is in the queue
            # NOTE: provide unified interface to mlp and dft
            if frames_to_work:
                self.worker.directory = self.directory/self.CALC_DIRNAME/f"gen{self.cur_gen}"
                _ = self.worker.run(frames_to_work) # retrieve later
        else:
            # --- update population
            self.pfunc("\n\n===== Update Population =====")
            # TODO: population settings
            #self._prepare_population(end_of_gen)
            if end_of_gen:
                # - create the population used for crossover and mutation
                current_population = Population(
                    data_connection = self.da,
                    population_size = self.pop_manager.gen_size,
                    comparator = self.comparing
                )
                # - 
                self.pop_manager._update_generation_settings(current_population, self.mutations, self.pairing)
                # - 
                current_candidates = self.pop_manager._prepare_current_population(
                    cur_gen=self.cur_gen, database=self.da, population=current_population, 
                    generator=self.generator, pairing=self.pairing, mutations=self.mutations
                )

                self.pfunc("\n\n===== Optimisation =====")
                # TODO: send candidates directly to worker that respects the batchsize
                frames_to_work = []
                for atoms in current_candidates:
                    self.pfunc("\n\n ----- start to run structure %s -----" %atoms.info["confid"])
                    frames_to_work.append(atoms)
                    self.da.mark_as_queued(atoms) # this marks relaxation is in the queue
                if frames_to_work:
                    self.worker.directory = self.directory/self.CALC_DIRNAME/f"gen{self.cur_gen}"
                    _ = self.worker.run(frames_to_work) # retrieve later
            else:
                self.pfunc("Current generation has not finished...")

        # --- check if there were finished jobs
        self.worker.directory = self.directory/self.CALC_DIRNAME/f"gen{self.cur_gen}"
        self.worker.inspect(resubmit=True)
        print(self.worker.directory)
        if self.worker.get_number_of_running_jobs() == 0:
            converged_candidates = self.worker.retrieve()
            for cand in converged_candidates:
                print(cand)
                print(cand.info)
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
        else:
            self.pfunc("Worker is unfinished.")

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
        #self.generator._print_blmin(self.blmin)

        return
    
    def is_converged(self):
        """ check whether the search is converged
        """
        max_gen = self.conv_dict["generation"]
        if self.cur_gen > max_gen and (self.num_relaxed_gen == self.num_unrelaxed_gen):
            return True
        else:
            return False
    
    def report(self):
        self.pfunc("restart the database...")
        self.__restart()
        results = self.directory/"results"
        if not results.exists():
            results.mkdir()

        # - write structures
        all_relaxed_candidates = self.da.get_all_relaxed_candidates()
        write(results/"all_candidates.xyz", all_relaxed_candidates)

        # - plot population evolution
        data = []
        cur_gen_num = self.da.get_generation_number() # equals finished generation plus one
        self.pfunc(f"Current generation number: {cur_gen_num}")
        for i in range(cur_gen_num):
            #print('generation ', i)
            energies = [
                atoms.get_potential_energy() for atoms in all_relaxed_candidates 
                    if atoms.info["key_value_pairs"]["generation"]==i
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
            fontweight="bold"
        )

        for i, energies in data:
            ax.scatter([i]*len(energies), energies)
        
        plt.savefig(results/"pop.png")

        return

    def refine(
        self, ref_worker
    ):
        """ refine structures with DFT (VASP)
            the number of structures is determined by the rule
        """
        self.pfunc("restart the database...")
        self.__restart()

        # - get all candidates
        results = self.directory / "results"
        if not results.exists():
            results.mkdir()
        all_relaxed_candidates = self.da.get_all_relaxed_candidates()
        sorted_candidates = sorted(
            all_relaxed_candidates, key=lambda atoms:atoms.info["key_value_pairs"]["raw_score"],
            reverse=True
        )
        nframes = len(sorted_candidates)

        # - selection
        from GDPy.selector import create_selector
        select_params = self.ga_dict.get("select", [])
        select_dpath = results/"select"
        if not select_dpath.exists():
            select_dpath.mkdir()
        selector = create_selector(select_params, directory=select_dpath)
        selector.pfunc = self.logger.info

        selected_frames = selector.select(sorted_candidates)
        nselected = len(selected_frames)
        self.pfunc(f"Find {nselected} frames for refinement...")

        # - computation
        ref_worker.directory = results/"refine"
        ref_worker.run(selected_frames)

        # NOTE: need last structure or entire trajectory?
        ref_worker.inspect()
        new_frames = ref_worker.retrieve()
        if new_frames:
            write(results/"refine"/"refined_candidates.xyz", new_frames, append=True)

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
        if "number_of_variable_cell_vectors" in params.keys():
            params["number_of_variable_cell_vectors"] = self.generator.number_of_variable_cell_vectors
        if "cellbounds" in params.keys():
            params["cellbounds"] = self.generator.cell_bounds
        if "test_dist_to_slab" in params.keys():
            params["test_dist_to_slab"] = self.generator.test_dist_to_slab
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
        mutation_list = op_dict.get("mutation", [])
        #self.pfunc(mutation_list)
        if not isinstance(mutation_list, list):
            mutation_list = [mutation_list]
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
            if "cellbounds" in params.keys():
                params["cellbounds"] = self.generator.cell_bounds
            if "number_of_variable_cell_vectors" in params.keys():
                params["number_of_variable_cell_vectors"] = self.generator.number_of_variable_cell_vectors
            if "used_modes_file" in params.keys():
                params["used_modes_file"] = self.directory/self.CALC_DIRNAME/"used_modes.json"
            # NOTE: check this mutation whether valid for this system
            if kwargs is None:
                prob = 1.0
            else:
                prob = kwargs.pop("prob", 1.0)
            probs.append(prob)
            if isinstance(kwargs, dict):
                params.update(**kwargs)
            mutations.append(mut(**params))

        self.pfunc("--- mutations ---")
        #self.pfunc(f"mutation probability: {self.pmut}")
        for mut, prob in zip(mutations, probs):
            self.pfunc(f"Use mutation {mut.descriptor} with prob {prob}.")
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

        starting_population = self.pop_manager._prepare_initial_population(generator=self.generator)

        self.pfunc(f"save population {len(starting_population)} to database")
        for a in starting_population:
            da.add_unrelaxed_candidate(a)
        
        # TODO: change this to the DB interface
        self.pfunc("save population size {0} into database...".format(self.pop_manager.gen_size))
        row = da.c.get(1)
        new_data = row["data"].copy()
        new_data["population_size"] = self.pop_manager.gen_size
        da.c.update(1, data=new_data)

        self.da = DataConnection(self.db_name)

        return
    
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
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            atoms.info["key_value_pairs"]["raw_score"] = -energy
            from ase.build import niggli_reduce
            from ase.calculators.singlepoint import SinglePointCalculator
            if self.generator.cell_bounds:
                stress = atoms.get_stress()
                niggli_reduce(atoms)
                calc = SinglePointCalculator(
                    atoms, energy=energy, forces=forces, stress=stress
                )
                atoms.calc = calc
                if self.generator.cell_bounds.is_within_bounds(atoms.get_cell()):
                    atoms.info["key_value_pairs"]["raw_score"] = -energy
                else:
                    atoms.info["key_value_pairs"]["raw_score"] = -1e8
        elif target == "barrier":
            pass
        else:
            raise RuntimeError(f"Unknown target {target}...")

        return

if __name__ == "__main__":
    pass
