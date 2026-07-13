from typing import List

from .serialization import save_model
from .sgp import SGP


class GaussianProcessTrainer:

    name = "gp"
    directory = None

    def __init__(
        self,
        config: dict,
        type_list: List[str] = None,
        train_epochs: int = 200,
        directory=".",
        command="train",
        freeze_command="freeze",
        random_seed: int = None,
        *args,
        **kwargs,
    ):
        self.directory = directory
        self.command = command
        self.freeze_command = freeze_command

    @property
    def frozen_name(self):
        return f"{self.name}.npz"

    def _resolve_freeze_command(self, *args, **kwargs):
        return ""

    def train(self, dataset, init_model=None, *args, **kwargs):
        frames = dataset.get("frames", dataset.get("structures", []))
        forces = dataset.get("forces", [])

        sgp = SGP(n_inducing=100, use_2b=True, use_3b=False)
        sgp.fit(frames, list(forces))

        import os
        import pathlib
        os.makedirs(str(self.directory), exist_ok=True)
        save_model(sgp, str(pathlib.Path(self.directory) / self.frozen_name))


# Self-register when this module is imported
try:
    from gdpx.trainer import REGISTER as _tr_reg
    _tr_reg.register(GaussianProcessTrainer)
except Exception:
    pass

    def write_input(self, dataset, *args, **kwargs):
        return

    def read_convergence(self) -> bool:
        return True
