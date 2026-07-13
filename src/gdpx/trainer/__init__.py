#!/usr/bin/env python3
# -*- coding: utf-8 -*


from gdpx import config
from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("trainer")

from .deepmd.deepmd import DeepmdTrainer

REGISTER.register(DeepmdTrainer)

try:
    from .deepmd.deepmd_jax import DeepmdJaxTrainer

    REGISTER.register(DeepmdJaxTrainer)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`deepmd_jax`':<16s} -> require `{e.name}`.")

from .nequip import NequipTrainer

REGISTER.register(NequipTrainer)

from .mace import MaceTrainer

REGISTER.register(MaceTrainer)

from .reann.beann import BeannTrainer

REGISTER.register(BeannTrainer)

from .reann.reann import ReannTrainer

REGISTER.register(ReannTrainer)

# GaussianProcessTrainer is registered via import hook in gdpx/potential/gp/trainer.py


if __name__ == "__main__":
    ...
