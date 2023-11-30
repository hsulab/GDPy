#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ase.ga.standardmutations import RattleMutation, PermutationMutation, MirrorMutation
from ase.ga.particle_mutations import (
    RandomMutation, RandomPermutation, COM2surfPermutation, Poor2richPermutation, 
    Rich2poorPermutation, SymmetricSubstitute, RandomSubstitute
)
from ase.ga.standardmutations import StrainMutation
from ase.ga.soft_mutation import SoftMutation, BondElectroNegativityModel

from gdpx.core.register import registers

# - standard
registers.builder.register("rattle")(RattleMutation)
registers.builder.register("permutation")(PermutationMutation)
registers.builder.register("mirror")(MirrorMutation)

# - bulk
registers.builder.register("strain")(StrainMutation)
registers.builder.register("soft")(SoftMutation)

# - cluster
#registers.builder.register("random")(RandomMutation)


if __name__ == "__main__":
    ...