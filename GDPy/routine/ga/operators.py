#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" here list operators can be import from ase.ga and custom ones
"""

import numpy as np

# --- comparator
# ofp_comparator: OFPComparator
# particle_comparator: NNMatComparator
# standard_comparators: InteratomicDistanceComparator,
#   SequentialComparator, StringComparator, EnergyComparator,
#   RawScoreComparator

from ase.ga.ofp_comparator import OFPComparator
from ase.ga.particle_comparator import NNMatComparator
from ase.ga.standard_comparators import InteratomicDistanceComparator

InteratomicDistanceComparator_params = dict(
    n_top=None, pair_cor_cum_diff=0.015,
    pair_cor_max=0.7, dE=0.02, mic=False
)

NNMatComparator_params = dict(
    d=0.2, elements=None, mic=False
)

OFPComparator_params = dict(
    n_top=None, dE=1, cos_dist_max=0.005,
    rcut=20, binwidth=0.05, sigma=0.02, nsigma=4,
    pbc=True, maxdims=None, recalculate=False
)

# --- crossover
# particle_crossovers: CutSpliceCrossover
# cutandsplicepairing: CutAndSplicePairing

from ase.ga.particle_crossovers import CutSpliceCrossover
from ase.ga.cutandsplicepairing import CutAndSplicePairing

# this is for particle (cluster) systems
CutSpliceCrossover_params = dict(
    blmin=None,
    keep_composition=True,
    rng=np.random
)

# for surface systems
CutAndSplicePairing_params = dict(
    slab=None, n_top=None, blmin=None,
    number_of_variable_cell_vectors=0, p1=1, p2=0.05,
    minfrac=None, cellbounds=None, test_dist_to_slab=True,
    use_tags=False, rng=np.random, verbose=False
)

# --- mutations
# standardmutation: Rattle, Permutation, Mirror, 
#   Strain, PermuStrain, Rotational, RotationalMutation
# particle_mutations: Random, RandomPermutation, COM2surfPermutation, 
#   Poor2richPermutation, Rich2poorPermutation, SymmetricSubstitute, RandomSubstitute
# soft_mutation: SoftMutation

from ase.ga.standardmutations import RattleMutation, PermutationMutation, MirrorMutation
from ase.ga.particle_mutations import (
    RandomMutation, RandomPermutation, COM2surfPermutation, Poor2richPermutation, 
    Rich2poorPermutation, SymmetricSubstitute, RandomSubstitute
)
from ase.ga.standardmutations import StrainMutation
from ase.ga.soft_mutation import SoftMutation, BondElectroNegativityModel

# --- standard
RattleMutation_params = dict(
    blmin=None, n_top=None, rattle_strength=0.8,
    rattle_prop=0.4, test_dist_to_slab=True, use_tags=False,
    verbose=False, rng=np.random
)

PermutationMutation_params = dict(
    n_top=None, probability=0.33, 
    test_dist_to_slab=True, use_tags=False, blmin=None,
    rng=np.random, verbose=False
)

MirrorMutation_params = dict(
    blmin=None, n_top=None, reflect=False, 
    rng=np.random, verbose=False
)

# --- bulk
StrainMutation_params = dict(
    blmin=None, cellbounds=None, stddev=0.7,
    number_of_variable_cell_vectors=3, use_tags=False,
    rng=np.random, verbose=False
)
SoftMutation_params = dict(
    blmin=None, bounds=[0.5, 2.0],
    calculator=BondElectroNegativityModel, rcut=10.,
    used_modes_file="used_modes.json", use_tags=False,
    verbose=False
)

# --- particles
# NOTE: some mutations may cause the missing of atoms.info such as confid
RandomMutation_params = dict(
    length=2, # maxstep to move in AA
    num_muts=1, # number of repetitive mutations
    rng=np.random
)

RandomPermutation_params = dict(
    elements=None, num_muts=1,
    rng=np.random
)