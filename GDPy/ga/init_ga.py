#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random
import pathlib
from typing import Union
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
        self.calc_dict = ga_dict['calculation']
        self.db_name = pathlib.Path(ga_dict['database'])
        return

    def run(self):
        """ main procedure
        """
        if not self.db_name.exists():
            print('create a new database...')
            self.create_surface()
            self.create_init_population()
        else:
            print('restart the database...')
            self._restart()
            if self.calc_dict['machine'] == 'slurm':
                # register machine and check jobs in virtual queue
                self.register_machine()
                self.pbs_run.check_status()
                # TODO: resubmit some jobs
                # try mutation and pairing
                self.register_operators()
                self.form_population()
                self.reproduce()
                # check convergence
            else:
                # local
                pass

        return
    
    def _initialise(self):

        return

    def _restart(self):
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

    def register_machine(self):
        tmp_folder = 'tmp_folder/'
        # The PBS queing interface is created
        self.pbs_run = SlurmQueueRun(
            self.da,
            tmp_folder=tmp_folder,
            job_prefix='goffee',
            n_simul=5,
            incar = self.calc_dict['incar']
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

        return
    
    def add_random_structures(self):
        """ add random structures into database
            can be done during any time in global optimisation
        """

        return

    def run_local_optimisation(self):
        """"""
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
                self.da.add_unrelaxed_candidate(a3, description=desc)

                if random() < mutation_probability:
                    a3_mut, desc = self.mutations.get_new_individual([a3])
                    if a3_mut is not None:
                        self.da.add_unrelaxed_step(a3_mut, desc)
                        a3 = a3_mut
                self.pbs_run.relax(a3)
                print('generate offspring a3 ', desc)
                break
            else:
                continue
        else:
            print('cannot generate offspring a3')

        return

if __name__ == "__main__":
    pass