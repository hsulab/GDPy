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

# Standard
registers.builder.register("rattle")(RattleMutation)
registers.builder.register("mirror")(MirrorMutation)

from .buffer import RattleBufferMutation
registers.builder.register("rattle_buffer")(RattleBufferMutation)

# Bulk
registers.builder.register("strain")(StrainMutation)
registers.builder.register("soft")(SoftMutation)

# Cluster
#registers.builder.register("random")(RandomMutation)


if __name__ == "__main__":
    ...
