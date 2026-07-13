import copy

from gdpx.potential.manager import BasePotentialManager

from .calculator import GPCalculator
from .serialization import load_model


class GaussianProcessManager(BasePotentialManager[GPCalculator]):

    name = "gp"
    implemented_backends = ("ase",)
    valid_combinations = (("ase", "ase"),)

    def register_calculator(self, calc_params: dict, **kwargs):
        super().register_calculator(calc_params, **kwargs)

        params = copy.deepcopy(calc_params)
        model_paths = params.pop("model", [])

        gp_model = load_model(model_paths[0])

        self.calc = GPCalculator(gp_model=gp_model)
