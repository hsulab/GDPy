#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random
import pathlib
from typing import Union
from ase.ga import population
import numpy as np

import ase.data

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

    def __init__(self, ga_dict: dict):
        """"""
        self.ga_dict = ga_dict
        self.db_name = pathlib.Path(ga_dict["database"])

        # settings for minimisation
        self.calc_dict = ga_dict["calculation"]
        self.machine = self.calc_dict["machine"]

        # mutation operators
        self.mutation_dict = ga_dict["mutation"]

        # check prefix, should not be too long since qstat cannot display
        #if len(self.calc_dict["prefix"]) > 6:
        #    raise ValueError("prefix is too long...")

        return

    def run(self):
        """ main procedure
        """
        # TODO: check database existence and generation number to determine restart
        if not self.db_name.exists():
            print("create a new database...")
            self.create_surface()
            self.create_init_population()
        else:
            print('restart the database...')
            self.__restart()
            if self.machine == "serial":
                # find z-axis constraint
                cons_maxidx = self.ga_dict["surface"].get("constraint", None)
                positions = self.slab.get_positions()
                self.zmin = np.max(positions[range(cons_maxidx),2])
                print("fixed atoms lower than {0} AA".format(self.zmin))

                # start minimisation
                self.__register_calculator()
                print("===== Initial Population =====")
                while (self.da.get_number_of_unrelaxed_candidates()):
                    atoms = self.da.get_an_unrelaxed_candidate()
                    print("start to run structure %s" %atoms.info["confid"])
                    self.worker.reset()
                    atoms.calc = self.worker
                    min_atoms, min_results = self.worker.minimise(
                        atoms,
                        **self.calc_dict["minimisation"],
                        zmin = self.zmin + 0.2
                    )
                    confid = atoms.info["confid"]
                    min_atoms.info['confid'] = confid
                    # add few information
                    min_atoms.info['data'] = {}
                    min_atoms.info['key_value_pairs'] = {'extinct': 0}
                    min_atoms.info['key_value_pairs']['raw_score'] = -min_atoms.get_potential_energy()
                    self.da.add_relaxed_step(min_atoms)
                    print(min_results)
                
                # start reproduce
                self.form_population()
                population_size = self.ga_dict['population']['init_size']
                max_gen = self.ga_dict["convergence"]["generation"]
                for ig in range(1,max_gen+1):
                    cur_gen = self.da.get_generation_number()
                    assert cur_gen == ig, "generation number not consistent!!! {0}!={1}".format(ig, cur_gen)
                    print("===== Generation {0} =====".format(cur_gen))
                    for j in range(population_size):
                        print("  offspring ", j)
                        self.reproduce()
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
                while (
                    self.pbs_run.number_of_jobs_running() + relaxed_num_strus_gen < population_size
                ):
                    self.reproduce()
                else:
                    print('enough jobs are running for current generation...')

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
                self.register_operators()
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

        all_atom_types = get_all_atom_types(self.slab, self.atom_numbers_to_optimize)
        self.blmin = closest_distances_generator(
            all_atom_types,
            ratio_of_covalent_radii=0.7
        )

        # mutation operators
        self.register_operators()

        return
    
    def report(self):
        print('restart the database...')
        self._restart()
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

    def make(self, num):
        print('restart the database...')
        self._restart()
        results = pathlib.Path.cwd() / 'results'
        if not results.exists():
            results.mkdir()
        all_relaxed_candidates = self.da.get_all_relaxed_candidates()
        sorted_candidates = sorted(
            all_relaxed_candidates, key=lambda atoms:atoms.info['key_value_pairs']['raw_score'],
            reverse=True
        )
        mosted = sorted_candidates[:num]
        print('Most %d Structures' %num)
        for atoms in mosted:
            print(atoms.info['confid'], 'raw_score: ', atoms.info['key_value_pairs']['raw_score'])
        write(results / ('most-%s.xyz' %num), mosted)

        from GDPy.ga.make_all_vasp import create_by_ase
        for atoms in mosted:
            dname = pathlib.Path.cwd() / 'accurate' / ('cand{0}'.format(atoms.info['confid']))
            create_by_ase(
                atoms, self.ga_dict["postprocess"]["incar"],
                dname
            )

        return
    
    def register_operators(self):
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
        self.mutations = OperationSelector(
            [1., 1., 1.], # probabilities for each mutation
            [
                RattleMutation(self.blmin, self.n_to_optimize),
                MirrorMutation(self.blmin, self.n_to_optimize),
                PermutationMutation(self.n_to_optimize)
            ] # operator list
        )

        return
    
    def __register_calculator(self):
        """register serial calculator"""
        from GDPy.calculator.reax import LMPMin
        self.worker = LMPMin(
            **self.calc_dict["kwargs"], 
            model_params = self.calc_dict["potential"]
        )

        return

    def register_machine(self):
        """register PBS/Slurm machine for computationally massive jobs"""
        tmp_folder = 'tmp_folder/'
        # The PBS queing interface is created
        self.pbs_run = SlurmQueueRun(
            self.da,
            tmp_folder=tmp_folder,
            n_simul=20,
            incar = self.calc_dict['incar'],
            prefix = self.calc_dict['prefix'] # TODO: move to input json
        )

        return

    def create_surface(self) -> None:
        # unpack info
        init_dict = self.ga_dict['surface']
        substrate = init_dict['substrate']
        surfsize = init_dict['surfsize']
        composition = init_dict['composition']
        constraint = init_dict.get('constraint', None)

        # create the surface
        self.slab = read(substrate)
        if constraint is not None:
            # TODO: convert index string to list
            self.slab.set_constraint(FixAtoms(indices=range(constraint)))

        # define the volume in which the adsorbed cluster is optimized
        # the volume is defined by a corner position (p0)
        # and three spanning vectors (v1, v2, v3)
        pos = self.slab.get_positions()
        cell = self.slab.get_cell()
        p0 = np.array([0., 0., max(pos[:, 2]) + surfsize[0]]) # origin of the box
        v1 = cell[0, :] * 1.0
        v2 = cell[1, :] * 1.0
        v3 = cell[2, :]
        v3[2] = surfsize[1]

        # output summary
        print("system cell", cell.complete())
        vec3_format = '{:>8.4f}  {:>8.4f}  {:>8.4f}\n'
        print("variation box")
        content = 'origin ' + vec3_format.format(*list(p0))
        content += 'xxxxxx ' + vec3_format.format(*list(v1))
        content += 'xxxxxx ' + vec3_format.format(*list(v2))
        content += 'xxxxxx ' + vec3_format.format(*list(v3))
        print(content)

        # Define the composition of the atoms to optimize
        self.atom_numbers = []
        for elem, num in composition.items():
            self.atom_numbers.extend([ase.data.atomic_numbers[elem]]*num)

        # define the closest distance two atoms of a given species can be to each other
        unique_atom_types = get_all_atom_types(self.slab, self.atom_numbers)
        blmin = closest_distances_generator(
            atom_numbers=unique_atom_types,
            ratio_of_covalent_radii=0.7
        )

        # create the starting population
        self.generator = StartGenerator(
            self.slab, self.atom_numbers, blmin,
            box_to_place_in=[p0, [v1, v2, v3]]
        ) # structure generator

        return 

    def create_init_population(
            self, 
        ):
        # unpack info
        population_size = self.ga_dict['population']['init_size']

        # generate the starting population
        starting_population = [self.generator.get_new_candidate() for i in range(population_size)]

        # create the database to store information in
        d = PrepareDB(
            db_file_name = self.db_name,
            simulation_cell = self.slab,
            stoichiometry = self.atom_numbers
        )

        print('save population to database')
        for a in starting_population:
            d.add_unrelaxed_candidate(a)
        
        # TODO: change this to the DB interface
        print("save population size {0} into database...".format(population_size))
        row = d.c.get(1)
        new_data = row['data'].copy()
        new_data['population_size'] = population_size
        d.c.update(1, data=new_data)

        return
    
    def add_random_structures(self):
        """ add random structures into database
            can be done during any time in global optimisation
        """

        return

    def run_local_optimisation(self):
        """
        This is for initial population optimisation
        """
        self.da = DataConnection(self.db_name)

        self.register_machine()
        # Relax all unrelaxed structures (e.g. the starting population)
        while (self.da.get_number_of_unrelaxed_candidates() and not self.pbs_run.enough_jobs_running()):
            a = self.da.get_an_unrelaxed_candidate()
            print('start to run structure %s' %a.info['confid'])
            self.pbs_run.relax(a)

        return
    
    def form_population(self):
        """"""
        # set current population
        # usually, it should be the same as the initial size
        # but for variat composition search, a large init size can be useful
        population_size = self.ga_dict['population']['init_size']
        # create the population
        self.population = Population(
            data_connection=self.da,
            population_size=population_size,
            comparator=self.comp
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
        mutation_probability = self.ga_dict['mutation']['prob']
        #while (not self.pbs_run.enough_jobs_running() and
        #       len(self.population.get_current_population()) > 2):
        a1, a2 = self.population.get_two_candidates()
        for i in range(10):
            # try 10 times
            a3, desc = self.pairing.get_new_individual([a1, a2])
            if a3 is not None:
                self.da.add_unrelaxed_candidate(a3, description=desc) # if mutation happens, it will not be relaxed

                mut_desc = ''
                if random() < mutation_probability:
                    a3_mut, mut_desc = self.mutations.get_new_individual([a3])
                    if a3_mut is not None:
                        self.da.add_unrelaxed_step(a3_mut, mut_desc)
                        a3 = a3_mut
                print('generate offspring a3 ', desc + ' ' + mut_desc)
                if self.machine == "serial":
                    print("start to run structure %s" %a3.info["confid"])
                    self.worker.reset()
                    a3.calc = self.worker
                    min_atoms, min_results = self.worker.minimise(
                        a3,
                        **self.calc_dict["minimisation"],
                        zmin = self.zmin + 0.2
                    )
                    confid = a3.info["confid"]
                    min_atoms.info['confid'] = confid
                    # add few information
                    min_atoms.info['data'] = {}
                    min_atoms.info['key_value_pairs'] = {'extinct': 0}
                    min_atoms.info['key_value_pairs']['raw_score'] = -min_atoms.get_potential_energy()
                    self.da.add_relaxed_step(min_atoms)
                    print(min_results)
                elif self.machine == "slurm":
                    self.pbs_run.relax(a3)
                else:
                    pass
                break
            else:
                continue
        else:
            print('cannot generate offspring a3 after {0} attempts'.format(10))

        return

if __name__ == "__main__":
    pass
