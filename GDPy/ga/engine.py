#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from random import random
import pathlib
from pathlib import Path
import importlib
import warnings

import numpy as np

import ase.data
import ase.formula

from ase import Atoms
from ase.io import read, write
from ase.ga.data import PrepareDB, DataConnection
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator # generate bond distance list
from ase.ga.utilities import get_all_atom_types # get system composition (both substrate and top)
from ase.constraints import FixAtoms

from ase.ga.population import Population

from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.standardmutations import MirrorMutation, RattleMutation, PermutationMutation
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

    # local optimisation directory
    CALC_DIRNAME = "tmp_folder"
    PREFIX = "cand"

    # reproduction and mutation
    MAX_REPROC_TRY = 1

    # TODO: Neighbor list and parametrization parameters to screen
    # candidates before relaxation can be added. Default is not to use.

    find_neighbors = None
    perform_parametrization = None

    use_tags = True # perform atomic or molecular based search
    test_dist_to_slab = True
    test_too_far = True

    # - cell settings
    #number_of_variable_cell_vectors=0,
    #box_to_place_in=box_to_place_in,
    #box_volume=None,
    #splits=None,
    #cellbounds=None,


    def __init__(self, ga_dict: dict):
        """"""
        self.ga_dict = ga_dict

        # - check system type
        from GDPy.builder.randomBuilder import RandomGenerator
        self.generator = RandomGenerator(ga_dict["system"])

        # --- database ---
        self.db_name = pathlib.Path(ga_dict["database"])

        # --- calculation ---
        from GDPy.computation.worker.worker import create_worker
        self.worker = create_worker(ga_dict["worker"])
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
        print("\nTarget of Global Optimisation is ", target)

        # --- convergence ---
        self.conv_dict = ga_dict["convergence"]

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
            print("----- create a new database -----")
            self._create_initial_population()
            # make calculation dir
            print("----- create a new tmp_folder -----")
            self.__initialise()
        else:
            print("restart the database...")
            # balh
            self.__restart()

        # mutation and comparassion operators
        print("\n\n===== register operators =====")
        self._register_operators()

        print("\n\n===== register population =====")
        # TODO: population settings
        self.form_population()

        # check current generation number
        print("\n\n===== Generation Info =====")

        cur_gen = self.da.get_generation_number()
        print("current generation number: ", cur_gen)
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

        print("number of relaxed in current generation: ", num_relaxed_gen)
        print(sorted(relaxed_confids))
        print("number of unrelaxed in current generation: ", num_unrelaxed_gen)
        print(sorted(unrelaxed_confids))

        # - run
        # --- initial population
        if cur_gen == 0:
            print("\n\n===== Initial Population Calculation =====")
            while (self.da.get_number_of_unrelaxed_candidates()): # NOTE: this uses GADB get_atoms which adds extra_info
                # calculate structures from init population
                atoms = self.da.get_an_unrelaxed_candidate()
                print("\n\n ----- start to run structure %s -----" %atoms.info["confid"])
                # NOTE: provide unified interface to mlp and dft
                _ = self.worker.run([atoms]) # retrieve later
                self.da.mark_as_queued(atoms) # this marks relaxation is in the queue

        if (
            # nunrelaxed_gen == 0
            #nrelaxed_gen == unrelaxed_gen == self.population_size
            num_unrelaxed_gen < self.population_size
            # TODO: check whether worker can accept new jobs
        ):
            # TODO: can be aggressive, reproduce when relaxed structures are available
            print("not enough unrelaxed candidates for generation %d and try to reproduce..." %cur_gen)
            print("number before reproduction: ", self.worker.get_number_of_running_jobs() + num_relaxed_gen)
            count = 0
            while (
                self.worker.get_number_of_running_jobs() + num_relaxed_gen < self.population_size
            ):
                atoms = self._reproduce()
                if atoms:
                    # run opt
                    print("\n\n ----- start to run structure %s -----" %atoms.info["confid"])
                    _ = self.worker.run([atoms]) # retrieve later
                    self.da.mark_as_queued(atoms) # this marks relaxation is in the queue
                    count += 1
            else:
                print(f"{count} candidates were reproduced in this run...")
                print("enough jobs are running for the current generation...")
        else:
            print("Enough candidates or not finished relaxing current generation...")

        # --- check if there were finished jobs
        converged_candidates = self.worker.retrieve()
        for cand in converged_candidates:
            # TODO: use tags
            # update extra info
            extra_info = dict(
                data = {},
                key_value_pairs = {"extinct": 0}
            )
            cand.info.update(extra_info)
            # evaluate raw score
            self.evaluate_candidate(cand)
            print("  add relaxed cand ", cand.info["confid"])
            print("  with raw_score {:.4f}".format(cand.info["key_value_pairs"]["raw_score"]))
            self.da.add_relaxed_step(
                cand,
                find_neighbors=self.find_neighbors,
                perform_parametrization=self.perform_parametrization
            )

        # --- check generation
        cur_gen = self.da.get_generation_number()
        #if cur_gen > max_gen and (num_relaxed_gen == num_unrelaxed_gen):
        print("current generation number: ", cur_gen)
        if cur_gen > max_gen:
            print("reach maximum generation...")
            self.report()

        return
    
    def __initialise(self):
        # get basic system information
        self.atom_numbers_to_optimize = self.da.get_atom_numbers_to_optimize()
        self.n_to_optimize = len(self.atom_numbers_to_optimize)
        self.slab = self.generator.slab

        return

    def __restart(self):
        """"""
        # basic system info
        self.da = DataConnection(self.db_name)

        # get basic system information
        self.atom_numbers_to_optimize = self.da.get_atom_numbers_to_optimize()
        self.n_to_optimize = len(self.atom_numbers_to_optimize)
        print("fxxk: ", self.n_to_optimize)
        self.slab = self.da.get_slab()

        # set bond list minimum
        init_dict = self.ga_dict["system"]
        covalent_ratio = init_dict.get("covalent_ratio", 0.8)

        all_atom_types = get_all_atom_types(self.slab, self.atom_numbers_to_optimize)
        self.blmin = closest_distances_generator(
            all_atom_types,
            ratio_of_covalent_radii=covalent_ratio
        )
        self.generator._print_blmin(self.blmin)

        return
    
    def report(self):
        print('restart the database...')
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
        print('Current generation number: ', cur_gen_num)
        for i in range(cur_gen_num):
            #print('generation ', i)
            energies = [
                atoms.get_potential_energy() for atoms in all_relaxed_candidates 
                    if atoms.info['key_value_pairs']['generation']==i
            ]
            print(energies)
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
        print('restart the database...')
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
                print(f"There are {new_number} structures in the range.")
                break
        else:
            print("All structures are in the energy range.")
        number = np.min([number, new_number])

        print(f"Select {number} structures out of {nframes}...")
        mosted = sorted_candidates[:number]
        #for atoms in mosted:
        #    print(atoms.info['confid'], 'raw_score: ', atoms.info['key_value_pairs']['raw_score'])
        print("energy range: ", mosted[0].get_potential_energy(), "  ", mosted[-1].get_potential_energy())
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
        print("create refinement directory...")
        from GDPy.utils.data import vasp_creator, vasp_collector
        incar_template = self.ga_dict["postprocess"]["incar"]
        # prefix = Path.cwd() / "accurate"
        prefix = Path("/mnt/scratch2/users/40247882/oxides/eann-main/train-all/m25r/ga-Pt322-fp")
        if not prefix.exists():
            prefix.mkdir()
        else:
            print("skip accurate...")

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
        #print("operators: ", operators)
        #comp_settings = op_dict.get(section, None)
        if settings is not None:
            # - get operator
            op_name = settings.get("name", None)
            assert op_name is not None, f"No op name is provided."
            kwargs = settings.get("kwargs", None)
        else:
            op_name = default_op
            kwargs = None

        #print(f"use {op_name}")
        op_obj = getattr(operators, op_name)
        default_params = getattr(operators, op_obj.__name__+"_params")

        return op_obj, default_params, kwargs
    
    def _register_operators(self):
        """ register various operators
            comparator, pairing, mutation
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

        print("--- comparator ---")
        print(f"Use comparator {comparator.__name__}.")

        # --- crossover
        #print("operators: ", operators)
        crossover, params, kwargs = self.__get_operator(
            operators, op_dict.get("crossover", None), 
            "CutAndSplicePairing"
        )
        if "slab" in params.keys():
            params["slab"] = self.slab
        if "n_top" in params.keys():
            params["n_top"] = self.n_to_optimize
        if "blmin" in params.keys():
            params["blmin"] = self.generator.blmin
        if "use_tags" in params.keys():
            params["use_tags"] = self.use_tags
        if isinstance(kwargs, dict):
            params.update(**kwargs)
        self.pairing = crossover(**params)

        print("--- crossover ---")
        print(f"Use crossover {crossover.__name__}.")

        # --- mutations
        mutations, probs = [], []
        mutation_list = op_dict.get("mutation", None)
        #print(mutation_list)
        if isinstance(mutation_list, list):
            for mut_settings in mutation_list:
                mut, params, kwargs = self.__get_operator(
                    operators, mut_settings, "RattleMutation"
                )
                if "n_top" in params.keys():
                    params["n_top"] = self.n_to_optimize
                if "blmin" in params.keys():
                    params["blmin"] = self.generator.blmin
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
                params["blmin"] = self.generator.blmin
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

        print("--- mutations ---")
        print("mutation probability: ", self.pmut)
        for mut in mutations:
            print(f"Use mutation {mut.descriptor}.")
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

        print("\n\n===== Initial Population Creation =====")
        # read seed structures
        if self.pop_init_seed is not None:
            print("----- try to add seed structures -----")
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
                    print(f"  add converged seed {i}")
                    da.add_relaxed_candidate(atoms)
                else:
                    # run opt
                    pass
        else:
            seed_size = 0

        # generate the starting population
        print("start to create initial population")
        starting_population = self.generator.run(self.population_size - seed_size)
        #print("start: ", starting_population)
        print(f"finished creating initial population...")

        print(f"save population {len(starting_population)} to database")
        for a in starting_population:
            da.add_unrelaxed_candidate(a)
        
        # TODO: change this to the DB interface
        print("save population size {0} into database...".format(self.population_size))
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

        # print out population info
        #frames = self.population.get_current_population()
        #print('current population size: ', len(frames))
        #for atoms in frames:
        #    n_paired = atoms.info.get('n_paired', None)
        #    looks_like = atoms.info.get('looks_like', None)
        #    print(atoms.info['confid'], ' -> ', n_paired, ' -> ', looks_like)

        return
    
    def _reproduce(self):
        """generate an offspring"""
        # Submit new candidates until enough are running
        a3 = None
        a1, a2 = self.population.get_two_candidates()
        for i in range(self.MAX_REPROC_TRY):
            # try 10 times
            a3, desc = self.pairing.get_new_individual([a1, a2]) # NOTE: this also adds key_value_pairs to a.info
            if a3 is not None:
                self.da.add_unrelaxed_candidate(
                    a3, description=desc # here, desc is used to add "pairing": 1 to database
                ) # if mutation happens, it will not be relaxed
                #print("a3: ", a3.info)

                mut_desc = ""
                if random() < self.pmut:
                    a3_mut, mut_desc = self.mutations.get_new_individual([a3])
                    #print("a3_mut: ", a3_mut.info)
                    #print("mut_desc: ", mut_desc)
                    if a3_mut is not None:
                        self.da.add_unrelaxed_step(a3_mut, mut_desc)
                        a3 = a3_mut
                print("generate offspring a3 ", desc + " " + mut_desc + " after ", i+1, " attempts..." )
                break
            else:
                mut_desc = ""
                print("failed generating offspring a3 ", desc + " " + mut_desc + " after ", i+1, " attempts..." )
        else:
            print("cannot generate offspring a3 after {0} attempts".format(self.MAX_REPROC_TRY))

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
