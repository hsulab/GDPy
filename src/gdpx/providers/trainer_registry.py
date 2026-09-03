from gdpx import config
from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("trainer")

from .deepmd.training.deepmd import DeepmdTrainer
REGISTER.register(DeepmdTrainer)

try:
    from .deepmd.training.deepmd_jax import DeepmdJaxTrainer
    REGISTER.register(DeepmdJaxTrainer)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`deepmd_jax`':<16s} -> require `{e.name}`.")

from .nequip.trainer import NequipTrainer
REGISTER.register(NequipTrainer)

from .mace.trainer import MaceTrainer
REGISTER.register(MaceTrainer)

from .reann.training.beann import BeannTrainer
REGISTER.register(BeannTrainer)

from .reann.training.reann import ReannTrainer
REGISTER.register(ReannTrainer)

from .nnp.trainer import NnpTrainer
REGISTER.register(NnpTrainer)
