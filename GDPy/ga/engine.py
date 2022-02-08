#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random
import pathlib
from pathlib import Path
import warnings
import numpy as np

import ase.data

from ase import Atoms
from ase.io import read, write
from ase.ga.data import PrepareDB, DataConnection
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator # generate bond distance list
from ase.ga.utilities import get_all_atom_types # get system composition (both substrate and top)
from ase.constraints import FixAtoms

from ase.ga.population import Population

from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.standardmutations import MirrorMutation, RattleMutation, PermutationMutation
from ase.ga.offspring_creator import OperationSelector

from GDPy.machine.gamachine import SlurmQueueRun

"""
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

    CALC_DIRNAME = "tmp_folder"

    MAX_REPROC_TRY = 10

    def __init__(self, ga_dict: dict):
        """"""
        self.ga_dict = ga_dict

        # check system type
        system_type = ga_dict["system"].get("type", None)
        if system_type in self.implemented_systems:
            self.system_type = system_type
        else:
            raise KeyError("Must declare system type for exploration [bulk, cluster, surface].")

        self.__parse_system_parameters(ga_dict)

        # check database
        self.db_name = pathlib.Path(ga_dict["database"])

        # settings for minimisation
        self.calc_dict = ga_dict["calculation"]
        self.machine = self.calc_dict["machine"]

        return
    
    def __parse_system_parameters(self, ga_dict):
        """ parse system-specific parameters
        """
        self.type_list = list(ga_dict["system"]["composition"].keys())

        # mutation operators
        self.mutation_dict = ga_dict["mutation"]

        return

    def run(
        self, spath: None
    ):
        """ main procedure
        """
        # TODO: check database existence and generation number to determine restart
        if not self.db_name.exists():
            print("create a new database...")
            self.__create_random_structure_generator()
            self.__create_initial_population()
            # make calculation dir
            self.tmp_folder = pathlib.Path.cwd() / self.CALC_DIRNAME
            self.tmp_folder.mkdir()
            print("create a new tmp_folder...")
            # read seed structures
            if spath is not None:
                print("----- try to add seed structures -----")
                frames = read(spath, ":")
                # NOTE: check force convergence and only add converged structures
                # check atom permutation
                for i, atoms in enumerate(frames):
                    # TODO: check atom order
                    atoms.info["description"] = "seed {}".format(i)
                    atoms.info["data"] = {}
                    atoms.info["key_value_pairs"] = {}
                    atoms.info["key_value_pairs"]["raw_score"] = -atoms.get_potential_energy()
                    if True: # force converged
                        print(f"  add converged seed {i}")
                        self.da.add_relaxed_candidate(atoms)
                    else:
                        # run opt
                        pass
        else:
            print("restart the database...")
            # balh
            self.tmp_folder = pathlib.Path.cwd() / self.CALC_DIRNAME
            self.__restart()
            # check current generation number
            cur_gen = self.da.get_generation_number()
            if self.machine == "serial":

                # start minimisation
                print("\n\n===== register calculator =====")
                self.__register_calculator()

                if cur_gen == 0:
                    print("===== Initial Population =====")
                    while (self.da.get_number_of_unrelaxed_candidates()):
                        # calculate structures from init population
                        atoms = self.da.get_an_unrelaxed_candidate()
                        print("\n\n ----- start to run structure %s -----" %atoms.info["confid"])
                        self.__run_local_optimisation(atoms)
                
                # start reproduce
                self.form_population()
                population_size = self.ga_dict["population"]["init_size"]
                max_gen = self.ga_dict["convergence"]["generation"]
                cur_gen = self.da.get_generation_number()
                for ig in range(cur_gen,max_gen+1): # TODO-2
                    #assert cur_gen == ig, "generation number not consistent!!! {0}!={1}".format(ig, cur_gen)
                    print("===== Generation {0} =====".format(ig))
                    relaxed_num_strus_gen = len(list(self.da.c.select('relaxed=1,generation=%d'%ig)))
                    print('number of relaxed in current generation: ', relaxed_num_strus_gen)
                    # TODO: check remain population
                    for j in range(relaxed_num_strus_gen, population_size):
                        print("  offspring ", j)
                        self.reproduce()
                
                # report results
                results = pathlib.Path.cwd() / 'results'
                if not results.exists():
                    results.mkdir()
                all_relaxed_candidates = self.da.get_all_relaxed_candidates()
                write(results / 'all_candidates.xyz', all_relaxed_candidates)
                print("finished!!!")
            elif self.machine == "slurm":
                # register machine and check jobs in virtual queue
                self.register_machine()
                self.pbs_run.check_status()
                # TODO: if generation one and no relaxed ones, run_init_optimisation
                # try mutation and pairing
                self.form_population()

                # TODO: check is the current population is full
                cur_gen_num = self.da.get_generation_number()
                print('generation number: ', cur_gen_num)

                if cur_gen_num == 0:
                    print("===== Initial Population =====")
                    while (self.da.get_number_of_unrelaxed_candidates()):
                        # calculate structures from init population
                        atoms = self.da.get_an_unrelaxed_candidate()
                        print("start to run structure %s" %atoms.info["confid"])
                        # TODO: provide unified interface to mlp and dft
                        #self.__run_local_optimisation(atoms)
                        self.pbs_run.relax(atoms)

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
                print('number of unrelaxed in current generation: ', relaxed_num_strus_gen)
                print('number of running jobs in current generation: ', cur_jobs_running)
                if relaxed_num_strus_gen == self.population_size:
                    # TODO: can be aggressive, reproduce when relaxed structures are available
                    print("finished current generation and try to reproduce...")
                    while (
                        self.pbs_run.number_of_jobs_running() + relaxed_num_strus_gen < population_size
                    ):
                        self.reproduce()
                    else:
                        print('enough jobs are running for current generation...')
                else:
                    print("not finished relaxing current generation...")

            else:
                # local
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
    
    def _initialise(self):

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
        all_atom_types = get_all_atom_types(self.slab, self.atom_numbers_to_optimize)
        self.blmin = closest_distances_generator(
            all_atom_types,
            ratio_of_covalent_radii=0.7
        )
        self.__print_blmin()

        # mutation operators
        self.__register_operators()

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
    
    def __register_operators(self):
        """ register various operators
            comparator, pairing, mutation
        """
        # set operators
        self.comp = InteratomicDistanceComparator(
            n_top = self.n_to_optimize,
            pair_cor_cum_diff = 0.015,
            pair_cor_max = 0.7,
            dE = 0.02,
            mic = False
        )
        self.pairing = CutAndSplicePairing(
            self.slab, self.n_to_optimize, self.blmin
        )

        op_classes = {
            "Rattle": RattleMutation,
            "Mirror": MirrorMutation,
            "Permutation": PermutationMutation
        }

        # TODO: expose to custom input file
        op_names = self.mutation_dict.get("ops", ["Rattle", "Mirror", "Permutation"])
        rel_probs = self.mutation_dict.get("probs", [1.]*len(op_names)) # relative 
        assert len(op_names) == len(rel_probs), "number of mutation operators and probs is not consistent..."
        if len(self.type_list) == 1 and "Permutation" in op_names:
            raise RuntimeError("Single element system cannot use PermutationMutation...")
        
        # output
        content = "\n\n===== register mutations =====\n"
        for op_name, rel_prob in zip(op_names, rel_probs):
            content += "  {}  {}\n".format(op_name, rel_prob)
        print(content)

        # register
        operators = []
        for op_name in op_names:
            if op_name == "Rattle":
                op = RattleMutation(self.blmin, self.n_to_optimize)
            elif op_name == "Mirror":
                op = MirrorMutation(self.blmin, self.n_to_optimize)
            elif op_name == "Permutation":
                op = PermutationMutation(self.n_to_optimize)
            operators.append(op)

        self.mutations = OperationSelector(rel_probs, operators)

        return
    
    def __register_calculator(self):
        """ register serial calculator and optimisation worker
        """
        model = self.calc_dict["potential"]["model"]
        if model == "lasp":
            from GDPy.calculator.lasp import LaspNN
            self.calc = LaspNN(**self.calc_dict["kwargs"])
        elif model == "eann": # and inteface to lammps
            from GDPy.calculator.lammps import Lammps
            self.calc = Lammps(
                **self.calc_dict["kwargs"],
                pair_style = self.calc_dict["potential"]
            )
        else:
            raise ValueError("Unknown potential to calculation...")
        
        interface = self.calc_dict["interface"]
        if interface == "ase":
            from GDPy.calculator.ase_interface import AseDynamics
            self.worker = AseDynamics(self.calc, directory=self.calc.directory)
            # use ase no need to recaclc constraint since atoms has one
            self.cons_indices = None
        else: 
            # find z-axis constraint
            self.cons_indices = None
            if self.system_type == "surface":
                constraint = self.ga_dict["system"]["substrate"]["constraint"]
                if constraint is not None:
                    index_group = constraint.split()
                    indices = []
                    for s in index_group:
                        r = [int(x) for x in s.split(":")]
                        indices.append([r[0]+1, r[1]]) # starts from 1
                self.cons_indices = ""
                for s, e in indices:
                    self.cons_indices += "{}:{} ".format(s, e)
                print("constraint indices: ", self.cons_indices)
        
            if interface == "lammps":
                from GDPy.calculator.lammps import LmpDynamics as dyn
                # use lammps optimisation
            elif interface == "lasp":
                from GDPy.calculator.lasp import LaspDynamics as dyn
            else:
                raise ValueError("Unknown interface to optimisation...")

            self.worker = dyn(
                self.calc, directory=self.calc.directory
            )

        return

    def __run_local_optimisation(self, atoms):
        """
        This is for initial population optimisation
        """
        # check database alive
        assert hasattr(self, "da") == True

        repeat = self.calc_dict["repeat"]

        # TODO: maybe move this part to evaluate_structure
        confid = atoms.info["confid"]
        self.worker.reset()
        # self.worker.directory = self.tmp_folder / ("cand" + str(confid))
        self.calc.directory = self.tmp_folder / ("cand" + str(confid))
        self.worker.set_output_path(self.calc.directory)

        print(f"\nStart minimisation maximum try {repeat} times...")
        for i in range(repeat):
            print("attempt ", i)
            # atoms.calc = self.worker # TODO:
            min_atoms, min_results = self.worker.minimise(
                atoms,
                **self.calc_dict["minimisation"],
                constraint = self.cons_indices # for lammps
            )
            print(min_results)
            min_atoms.info['confid'] = confid
            # add few information
            min_atoms.info['data'] = {}
            min_atoms.info['key_value_pairs'] = {'extinct': 0}
            min_atoms.info['key_value_pairs']['raw_score'] = -min_atoms.get_potential_energy()
            maxforce = np.max(np.fabs(min_atoms.get_forces(apply_constraint=True)))
            if maxforce < self.calc_dict["minimisation"]["fmax"]:
                self.da.add_relaxed_step(min_atoms)
                break
            else:
                atoms = min_atoms
        else:
            # TODO: !!!
            self.da.add_relaxed_step(min_atoms)
            warnings.warn(f"Not converged after {repeat} minimisations, and save the last atoms...", UserWarning)

        return

    def register_machine(self):
        """register PBS/Slurm machine for computationally massive jobs"""
        # The PBS queing interface is created
        self.pbs_run = SlurmQueueRun(
            self.da,
            tmp_folder=self.CALC_DIRNAME,
            n_simul=20, # TODO: change this to population size
            incar = self.calc_dict['incar'],
            prefix = self.calc_dict['prefix'] # TODO: move to input json
        )
        # constraint comes along with self.slab when writing vasp-POSCAR

        return

    def __create_random_structure_generator(self) -> None:
        """ create a random structure generator
        """
        # unpack info
        init_dict = self.ga_dict["system"]
        composition = init_dict['composition']

        if self.system_type == "bulk":
            # TODO: specific routine for bulks
            pass
        elif self.system_type == "cluster":
            cell = np.array(init_dict["lattice"])
            self.slab = Atoms(cell = cell, pbc=True)
            self.cell_centre = np.sum(0.5*cell, axis=1)
            
            # set box to explore
            box_cell = np.array(init_dict["space"])
            p0 = np.zeros(3)
            #p0 = np.sum(0.5*cell, axis=1) # centre of the cell
            v1 = box_cell[0, :] 
            v2 = box_cell[1, :] 
            v3 = box_cell[2, :]

            # parameters
            box_to_place_in = [p0, [v1, v2, v3]]
            test_dist_to_slab = False
            test_too_far = False

        elif self.system_type == "surface":
            # read substrate
            substrate_dict = init_dict["substrate"]
            substrate_file = substrate_dict["file"]
            surfsize = substrate_dict["surfsize"]
            constraint = substrate_dict.get("constraint", None)

            # create the surface
            self.slab = read(substrate_file)
            if constraint is not None:
                index_group = constraint.split()
                indices = []
                for s in index_group:
                    r = [int(x) for x in s.split(":")]
                    indices.extend(list(range(r[0], r[1])))
                print(indices)
                self.slab.set_constraint(FixAtoms(indices=indices))

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
        print(self.slab)

        # Define the composition of the atoms to optimize
        atom_numbers = []
        for elem, num in composition.items():
            atom_numbers.extend([ase.data.atomic_numbers[elem]]*num)
        self.atom_numbers_to_optimize = atom_numbers
        unique_atom_types = get_all_atom_types(self.slab, atom_numbers)

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
            self.atom_numbers_to_optimize, # blocks
            blmin,
            number_of_variable_cell_vectors=0,
            box_to_place_in=box_to_place_in,
            box_volume=None,
            splits=None,
            cellbounds=None,
            test_dist_to_slab = test_dist_to_slab,
            test_too_far = test_too_far
        ) # structure generator

        return 

    def __create_initial_population(
            self, 
        ):
        # unpack info
        population_size = self.ga_dict['population']['init_size']

        # generate the starting population
        print("start to create initial population")
        nfailed = 0
        starting_population = []
        while len(starting_population) < population_size:
            maxiter = 100
            candidate = self.generator.get_new_candidate(maxiter=maxiter)
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
        print(f"Finished creating initial population with {nfailed} attempts...")

        # create the database to store information in
        da = PrepareDB(
            db_file_name = self.db_name,
            simulation_cell = self.slab,
            stoichiometry = self.atom_numbers_to_optimize
        )

        print('save population to database')
        for a in starting_population:
            da.add_unrelaxed_candidate(a)
        
        # TODO: change this to the DB interface
        print("save population size {0} into database...".format(population_size))
        row = da.c.get(1)
        new_data = row['data'].copy()
        new_data['population_size'] = population_size
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
        population_size = self.ga_dict['population']['init_size']
        # create the population
        self.population = Population(
            data_connection = self.da,
            population_size = population_size,
            comparator = self.comp
        )
        self.population_size = population_size

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
        mutation_probability = self.mutation_dict["pmut"]

        #while (not self.pbs_run.enough_jobs_running() and
        #       len(self.population.get_current_population()) > 2):
        a1, a2 = self.population.get_two_candidates()
        for i in range(self.MAX_REPROC_TRY):
            # try 10 times
            a3, desc = self.pairing.get_new_individual([a1, a2])
            if a3 is not None:
                self.da.add_unrelaxed_candidate(a3, description=desc) # if mutation happens, it will not be relaxed

                mut_desc = ""
                if random() < mutation_probability:
                    a3_mut, mut_desc = self.mutations.get_new_individual([a3])
                    if a3_mut is not None:
                        self.da.add_unrelaxed_step(a3_mut, mut_desc)
                        a3 = a3_mut
                print("generate offspring a3 ", desc + " " + mut_desc + " after ", i+1, " attempts..." )

                # run opt
                if self.machine == "serial":
                    print("start to run structure %s" %a3.info["confid"])
                    self.__run_local_optimisation(a3)
                elif self.machine == "slurm":
                    self.pbs_run.relax(a3)
                else:
                    pass
                break
            else:
                continue
        else:
            print('cannot generate offspring a3 after {0} attempts'.format(self.MAX_REPROC_TRY))

        return

if __name__ == "__main__":
    pass
