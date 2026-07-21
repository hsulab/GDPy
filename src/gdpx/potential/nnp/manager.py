import copy

from ..manager import BasePotentialManager


class NnAcsfManager(BasePotentialManager):

    name = "nnp"
    implemented_backends = ("ase",)
    valid_combinations = (("ase", "ase"),)

    def __init__(self):
        super().__init__()

    def register_calculator(self, calc_params, *args, **kwargs):
        super().register_calculator(calc_params, *args, **kwargs)
        from .calculator import ACSFNN
        self.calc = ACSFNN(**calc_params)

    def as_dict(self):
        return {"name": self.name, "params": copy.deepcopy(self.calc_params)}
