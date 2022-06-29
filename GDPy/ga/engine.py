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

    implemented_systems = ["bulk", "cluster", "surface"]

    supported_calculators = ["vasp", "lammps", "lasp"]

    # local optimisation directory
    CALC_DIRNAME = "tmp_folder"
    PREFIX = "cand"

    # reproduction and mutation
    MAX_REPROC_TRY = 10
    MAX_RANDOM_TRY = 100

    # default settings
    calc_dict = {
        "machine": "slurm", # serial(local), slurm, pbs
        "potential": "vasp", # vasp, lasp, eann, dp
        "kwargs": None, # input parameters
    }

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

        # check system type
        system_type = ga_dict["system"].get("type", None)
        if system_type in self.implemented_systems:
            self.system_type = system_type
        else:
            raise KeyError("Must declare system type for exploration [bulk, cluster, surface].")

        self.type_list = list(ga_dict["system"]["composition"].keys())

        # --- database ---
        self.db_name = pathlib.Path(ga_dict["database"])

        # --- calculation ---
        self.calc_dict = ga_dict["calculation"]
        self.machine = self.calc_dict["machine"]

        # --- directory ---
        prefix = self.calc_dict.get("prefix", "cand")
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
            self.__create_random_structure_generator()
            self.__create_initial_population()
            # make calculation dir
            print("----- create a new tmp_folder -----")
            self.tmp_folder = pathlib.Path.cwd() / self.CALC_DIRNAME
            self.tmp_folder.mkdir()

            self.__initialise()
        else:
            print("restart the database...")
            # balh
            self.tmp_folder = pathlib.Path.cwd() / self.CALC_DIRNAME
            self.__create_random_structure_generator()
            self.__restart()

        # mutation and comparassion operators
        print("\n\n===== register operators =====")
        self.__register_operators()

        print("\n\n===== register calculator =====")
        self.__register_calculator()
            
        # TODO: population settings
        self.form_population()

        # check current generation number
        print("\n\n===== Generation Info =====")

        cur_gen = self.da.get_generation_number()
        print("current generation number: ", cur_gen)
        max_gen = self.conv_dict["generation"]

        # output a few info
        unrelaxed_strus_gen = list(self.da.c.select("relaxed=0,generation=%d" %cur_gen))
        unrelaxed_confids = [row["gaid"] for row in unrelaxed_strus_gen]
        num_unrelaxed_gen = len(unrelaxed_confids)

        relaxed_strus_gen = list(self.da.c.select("relaxed=1,generation=%d" %cur_gen))
        relaxed_confids = [row["gaid"] for row in relaxed_strus_gen]
        num_relaxed_gen = len(relaxed_confids)

        print("number of relaxed in current generation: ", num_relaxed_gen)
        print(sorted(relaxed_confids))
        print("number of unrelaxed in current generation: ", num_unrelaxed_gen)
        print(sorted(unrelaxed_confids))

        # ===== create initial population =====

        # start generation and calculation
        if self.machine == "serial":
            # start minimisation
            if cur_gen == 0:
                print("\n\n===== Initial Population Calculation =====")
                while (self.da.get_number_of_unrelaxed_candidates()):
                    # calculate structures from init population
                    atoms = self.da.get_an_unrelaxed_candidate()
                    print("\n\n ----- start to run structure confid %s -----" %atoms.info["confid"])
                    self.__run_local_optimisation(atoms)
            
            # start reproduce
            cur_gen = self.da.get_generation_number()
            for ig in range(cur_gen, max_gen+1): # TODO-2
                #assert cur_gen == ig, "generation number not consistent!!! {0}!={1}".format(ig, cur_gen)
                print("\n\n===== Generation {0} =====".format(ig))
                num_relaxed_gen = len(list(self.da.c.select("relaxed=1,generation=%d" %ig)))
                print("number of relaxed in current generation: ", num_relaxed_gen)
                # TODO: check remain population
                # reproduce structures
                while num_relaxed_gen < self.population_size - self.pop_ran_size:
                    print(f"\n --- offspring {num_relaxed_gen} ---")
                    self.reproduce()
                    num_relaxed_gen = len(list(self.da.c.select("relaxed=1,generation=%d" %ig)))
                # random structure
                if self.pop_ran_size > 0:
                    print("generate random structures for this generation...")
                    nfailed, ran_candidates = self.__generate_random_structures(self.pop_ran_size)
                    print(f"finished creating random population with {nfailed} failed attempts...")
                    for i, candidate in enumerate(ran_candidates):
                        print(f"\n --- random {i} ---")
                        candidate.info["data"] = {}
                        candidate.info["key_value_pairs"] = {}
                        self.da.add_unrelaxed_candidate(candidate, description="random: RandomCandidate") # TODO: may change dataase object
                        print("\n\n ----- start to run structure confid %s -----" %candidate.info["confid"])
                        self.__run_local_optimisation(candidate)
            
            # report results
            self.report()
            print("finished properly!!!")

        elif self.machine == "slurm":
            print("number of running jobs in current generation: ", self.worker.number_of_jobs_running())
            print("\n\n===== Generation {0} =====".format(cur_gen))
            # check status
            converged_candidates = self.worker.check_status()
            for cand in converged_candidates:
                # evaluate raw score
                self.evaluate_candidate(cand)
                print("  add relaxed cand ", cand.info["confid"])
                print("  with raw_score {:.4f}".format(cand.info["key_value_pairs"]["raw_score"]))
                self.dc.add_relaxed_step(
                    cand,
                    find_neighbors=self.find_neighbors,
                    perform_parametrization=self.perform_parametrization
                )

            # check initial population
            if cur_gen == 0:
                print("\n\n===== Initial Population Calculation =====")
                while (self.da.get_number_of_unrelaxed_candidates()): # NOTE: this uses GADB get_atoms which adds extra_info
                    # calculate structures from init population
                    atoms = self.da.get_an_unrelaxed_candidate()
                    print("\n\n ----- start to run structure %s -----" %atoms.info["confid"])
                    # NOTE: provide unified interface to mlp and dft
                    self.__run_local_optimisation(atoms)

            if cur_gen > max_gen:
                print("reach maximum generation...")
                self.report()
                exit()

            # reproduce
            if (
                # nunrelaxed_gen == 0
                #nrelaxed_gen == unrelaxed_gen == self.population_size
                num_unrelaxed_gen < self.population_size
            ):
                # TODO: can be aggressive, reproduce when relaxed structures are available
                print("not enough unrelaxed candidates for generation %d and try to reproduce..." %cur_gen)
                print("number before reproduction: ", self.worker.number_of_jobs_running() + num_relaxed_gen)
                count = 0
                while (
                    self.worker.number_of_jobs_running() + num_relaxed_gen < self.population_size
                ):
                    self.reproduce()
                    count += 1
                else:
                    print(f"{count} candidates were reproduced in this run...")
                    print("enough jobs are running for current generation...")
            else:
                print("not finished relaxing current generation...")
        else:
            pass

        return
    
    def check_status(self):
        """"""
        if self.db_name.exists():
            print('restart the database...')
            self._restart()
            if self.calc_dict['machine'] == 'slurm':
                # register machine and check jobs in virtual queue
                self.register_machine()
                self.pbs_run.check_status()
                exit()
                # TODO: resubmit some jobs
                # try mutation and pairing
                self.__register_operators()
                self.form_population()
                # TODO: check is the current population is full
                cur_gen_num = self.da.get_generation_number()
                print('generation number: ', cur_gen_num)

                max_gen = self.ga_dict['convergence']['generation']
                if cur_gen_num > max_gen:
                    print('reach maximum generation...')
                    exit()

                #print(len(self.da.get_all_relaxed_candidates_after_generation(cur_gen_num)))
                unrelaxed_num_strus_gen = len(list(self.da.c.select('unrelaxed=1,generation=%d'%cur_gen_num)))
                relaxed_num_strus_gen = len(list(self.da.c.select('relaxed=1,generation=%d'%cur_gen_num)))
                population_size = self.ga_dict['population']['init_size']
                cur_jobs_running = self.pbs_run.number_of_jobs_running()
                print('number of relaxed in current generation: ', relaxed_num_strus_gen)
                print('number of running jobs in current generation: ', cur_jobs_running)
                #while (
                #    self.pbs_run.number_of_jobs_running() + relaxed_num_strus_gen < population_size
                #):
                #    self.reproduce()
                #else:
                #    print('enough jobs are running for current generation...')
        else:
            print("The database has not been initialised...")

        return
    
    def __initialise(self):
        # get basic system information
        self.atom_numbers_to_optimize = self.da.get_atom_numbers_to_optimize()
        self.n_to_optimize = len(self.atom_numbers_to_optimize)

        return

    def __restart(self):
        """"""
        # basic system info
        self.da = DataConnection(self.db_name)

        # get basic system information
        self.atom_numbers_to_optimize = self.da.get_atom_numbers_to_optimize()
        self.n_to_optimize = len(self.atom_numbers_to_optimize)
        self.slab = self.da.get_slab()

        # set bond list minimum
        init_dict = self.ga_dict["system"]
        covalent_ratio = init_dict.get("covalent_ratio", 0.8)

        all_atom_types = get_all_atom_types(self.slab, self.atom_numbers_to_optimize)
        self.blmin = closest_distances_generator(
            all_atom_types,
            ratio_of_covalent_radii=covalent_ratio
        )
        self.__print_blmin()

        return
    
    def __print_blmin(self):
        """"""
        elements = get_all_atom_types(self.slab, self.atom_numbers_to_optimize)
        nelements = len(elements)
        index_map = {}
        for i, e in enumerate(elements):
            index_map[e] = i
        distance_map = np.zeros((nelements, nelements))
        for (i, j), dis in self.blmin.items():
            distance_map[index_map[i], index_map[j]] = dis

        symbols = [ase.data.chemical_symbols[e] for e in elements]

        content =  "----- Bond Distance Minimum -----\n"
        content += " "*4+("{:>6}  "*nelements).format(*symbols) + "\n"
        for i, s in enumerate(symbols):
            content += ("{:<4}"+"{:>8.4f}"*nelements+"\n").format(s, *list(distance_map[i]))
        content += "note: default too far tolerance is 2 times\n"
        print(content)

        return
    
    def report(self):
        print('restart the database...')
        self.__restart()
        results = pathlib.Path.cwd() / 'results'
        if not results.exists():
            results.mkdir()
        all_relaxed_candidates = self.da.get_all_relaxed_candidates()
        write(results / 'all_candidates.xyz', all_relaxed_candidates)

        #for atoms in all_relaxed_candidates:
        #    print(atoms.info['key_value_pairs']['generation'])

        # plot population evolution
        data = []
        cur_gen_num = self.da.get_generation_number()
        print('Current generation number: ', cur_gen_num)
        for i in range(cur_gen_num+1):
            #print('generation ', i)
            energies = [
                atoms.get_potential_energy() for atoms in all_relaxed_candidates 
                    if atoms.info['key_value_pairs']['generation']==i
            ]
            print(energies)
            data.append([i, energies])
        
        import matplotlib as mpl
        mpl.use('Agg') #silent mode
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
        ax.set_title(
            'Population Evolution', 
            fontsize=20, 
            fontweight='bold'
        )

        for i, energies in data:
            ax.scatter([i]*len(energies), energies)
        
        plt.savefig(results/'pop.png')

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
    
    def __register_operators(self):
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
            params["blmin"] = self.blmin
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

        print("--- mutations ---")
        print("mutation probability: ", self.pmut)
        for mut in mutations:
            print(f"Use mutation {mut.descriptor}.")
        self.mutations = OperationSelector(probs, mutations, rng=np.random)

        return
    
    def __register_calculator(self):
        """ register serial calculator and optimisation worker
        """
        # NOTE: vasp not test
        from GDPy.potential.manager import create_pot_manager
        pm = create_pot_manager(self.calc_dict["potential"])
        self.dyn_params = self.calc_dict["dynamics"]
        self.worker = pm.create_worker(self.dyn_params)
        
        # NOTE: constraint, py convention
        self.cons_indices = None
        if self.system_type == "surface":
            constraint = self.ga_dict["system"]["substrate"]["constraint"]
            #if constraint is not None:
            #    index_group = constraint.split()
            #    indices = []
            #    for s in index_group:
            #        r = [int(x) for x in s.split(":")]
            #        indices.append([r[0]+1, r[1]]) # starts from 1
            #self.cons_indices = ""
            #for s, e in indices:
            #    self.cons_indices += "{}:{} ".format(s, e)
            self.cons_indices = constraint
            print("constraint indices: ", self.cons_indices)

        return

    def __run_local_optimisation(self, atoms):
        """ perform local optimisation
        """
        # check database alive
        assert hasattr(self, "da") == True

        if self.machine == "slurm":
            # TODO: move queue methods here
            self.worker.relax(atoms)
            return 
        cur_tags = atoms.get_tags().copy()
        print("tags before:", atoms.get_tags())

        # TODO: maybe move this part to evaluate_structure
        confid = atoms.info["confid"]
        self.worker.reset()
        cur_dir = self.tmp_folder / (self.PREFIX + str(confid))
        self.worker.set_output_path(cur_dir)

        # prepare extra info
        extra_info = {}
        extra_info["confid"] = confid
        extra_info["data"] = {}
        extra_info["key_value_pairs"] = {"extinct": 0}

        # run minimisation
        # NOTE: dynamics should maintain tags info
        min_atoms = self.worker.minimise(
            atoms,
            extra_info = extra_info,
            **self.dyn_params,
            constraint = self.cons_indices # for lammps and lasp
        )
        print("tags after:", min_atoms.get_tags())
        min_atoms.set_tags(cur_tags)

        # evaluate structure
        self.evaluate_candidate(min_atoms)

        # add relaxed candidate into database
        self.da.add_relaxed_step(min_atoms)

        return

    def __create_random_structure_generator(self) -> None:
        """ create a random structure generator
        """
        # unpack info
        init_dict = self.ga_dict["system"]

        if self.system_type == "bulk":
            # TODO: specific routine for bulks
            pass
        elif self.system_type == "cluster":
            cell = init_dict.get("lattice", None)
            if cell is None:
                cell = np.ones(3)*20.
            else:
                cell = np.array(cell)

            self.slab = Atoms(cell = cell, pbc=True)
            self.cell_centre = np.sum(0.5*cell, axis=1)
            
            # set box to explore
            # NOTE: shape (4,3), origin+3directions
            box_cell = init_dict.get("space", None)
            if box_cell is None:
                box_cell = np.zeros((4,3))
                box_cell[1:,:] = 0.5*cell
            else:
                box_cell = np.array(init_dict["space"])
            p0 = box_cell[0, :] 
            v1 = box_cell[1, :] 
            v2 = box_cell[2, :] 
            v3 = box_cell[3, :]

            # parameters
            box_to_place_in = [p0, [v1, v2, v3]]
            self.test_dist_to_slab = False
            self.test_too_far = False

        elif self.system_type == "surface":
            # read substrate
            substrate_dict = init_dict["substrate"]
            substrate_file = substrate_dict["file"]
            surfsize = substrate_dict["surfsize"]
            constraint = substrate_dict.get("constraint", None)

            # create the surface
            self.slab = read(substrate_file)
            # TODO: parse constraint directly to dynamics or just add constraint to atoms?
            #if constraint is not None:
            #    index_group = constraint.split()
            #    indices = []
            #    for s in index_group:
            #        r = [int(x) for x in s.split(":")]
            #        indices.extend(list(range(r[0], r[1])))
            #    print(indices)
            #    self.slab.set_constraint(FixAtoms(indices=indices))

            # define the volume in which the adsorbed cluster is optimized
            # the volume is defined by a corner position (p0)
            # and three spanning vectors (v1, v2, v3)
            pos = self.slab.get_positions()
            if init_dict["lattice"] is None:
                cell = self.slab.get_cell()
                cell = cell.complete() 
            else:
                cell = np.array(init_dict["lattice"])
            
            # create box for atoms to explore
            box_cell = init_dict.get("space", None)
            if box_cell is None:
                p0 = np.array([0., 0., np.max(pos[:, 2]) + surfsize[0]]) # origin of the box
                v1 = cell[0, :]
                v2 = cell[1, :]
                v3 = cell[2, :]
                v3[2] = surfsize[1]
                box_to_place_in = [p0, [v1, v2, v3]]
            else:
                box_cell = np.array(box_cell)
                if box_cell.shape[0] == 3:
                    # auto add origin for [0, 0, 0]
                    pass
                elif box_cell.shape[0] == 4:
                    box_to_place_in = box_cell

            # two parameters
            test_dist_to_slab = True
            test_too_far = True

        # output summary
        print("system cell", cell)
        vec3_format = '{:>8.4f}  {:>8.4f}  {:>8.4f}\n'
        print("variation box")
        content =  "origin " + vec3_format.format(*list(p0))
        content += "xxxxxx " + vec3_format.format(*list(v1))
        content += "xxxxxx " + vec3_format.format(*list(v2))
        content += "xxxxxx " + vec3_format.format(*list(v3))
        print(content)
        #print(self.slab)

        # --- Define the composition of the atoms to optimize ---
        composition = init_dict["composition"]
        blocks = [(k,v) for k,v in composition.items()] # for start generator
        for k, v in blocks:
            if k not in ase.data.chemical_symbols:
                print(f"Found chemical formula {k}")
                self.use_tags = True
                break
        else:
            print("Perform atomic search...")

        atom_numbers = []
        for species, num in composition.items():
            numbers = []
            for s, n in ase.formula.Formula(species).count().items():
                numbers.extend([ase.data.atomic_numbers[s]]*n)
            atom_numbers.extend(numbers*num)
        self.atom_numbers_to_optimize = atom_numbers
        unique_atom_types = get_all_atom_types(self.slab, atom_numbers)
        print(self.atom_numbers_to_optimize)
        print(blocks)

        # define the closest distance two atoms of a given species can be to each other
        covalent_ratio = init_dict.get("covalent_ratio", 0.8)
        print("colvent ratio is: ", covalent_ratio)
        blmin = closest_distances_generator(
            atom_numbers=unique_atom_types,
            ratio_of_covalent_radii = covalent_ratio # be careful with test too far
        )
        self.blmin = blmin

        print("neighbour distance restriction")
        self.__print_blmin()

        # create the starting population
        self.generator = StartGenerator(
            self.slab, 
            blocks, # blocks
            blmin,
            number_of_variable_cell_vectors=0,
            box_to_place_in=box_to_place_in,
            box_volume=None,
            splits=None,
            cellbounds=None,
            test_dist_to_slab = self.test_dist_to_slab,
            test_too_far = self.test_too_far,
            rng = np.random
        ) # structure generator

        return 
        
    def __generate_random_structures(self, ran_size):
        """"""
        nfailed = 0
        starting_population = []
        while len(starting_population) < ran_size:
            candidate = self.generator.get_new_candidate(maxiter=self.MAX_RANDOM_TRY)
            # TODO: add some geometric restriction here
            if candidate is None:
                # print(f"This creation failed after {maxiter} attempts...")
                nfailed += 1
            else:
                if self.system_type == "cluster":
                    com = candidate.get_center_of_mass().copy()
                    candidate.positions += self.cell_centre - com
                starting_population.append(candidate)
            #print("now we have ", len(starting_population))

        return nfailed, starting_population

    def __create_initial_population(
            self, 
        ):
        # create the database to store information in
        # TODO: move this part to where before generator is created
        da = PrepareDB(
            db_file_name = self.db_name,
            simulation_cell = self.slab,
            stoichiometry = self.atom_numbers_to_optimize
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
        nfailed, starting_population = self.__generate_random_structures(
            self.population_size - seed_size
        )
        print(f"finished creating initial population with {nfailed} failed attempts...")

        print("save population to database")
        for a in starting_population:
            da.add_unrelaxed_candidate(a)
        
        # TODO: change this to the DB interface
        print("save population size {0} into database...".format(self.population_size))
        row = da.c.get(1)
        new_data = row['data'].copy()
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
    
    def reproduce(self):
        """generate an offspring"""
        # Submit new candidates until enough are running
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

                # run opt
                print("\n\n ----- start to run structure confid %s -----" %a3.info["confid"])
                self.__run_local_optimisation(a3)

                break
            else:
                continue
        else:
            print("cannot generate offspring a3 after {0} attempts".format(self.MAX_REPROC_TRY))

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
            atoms.info["key_value_pairs"]["raw_score"] = -atoms.get_potential_energy()
        elif target == "barrier":
            pass
        else:
            raise RuntimeError(f"Unknown target {target}...")

        return

if __name__ == "__main__":
    pass
