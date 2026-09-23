import pathlib

from ase.calculators.eam import EAM

from gdpx.backend.ase import DummyCalculator
from gdpx.computation.lammps import Lammps

from .manager import BasePotentialManager
from .utils import canonicalise_input_models

EAM_FLAVOURS = (
    "eam",
    "eam/alloy",
    "eam/cd",
    "eam/fs",
    "eam/he",
)


class EamManager(BasePotentialManager):
    name = "eam"

    implemented_backends = (
        "ase",
        "lammps",
    )
    valid_combinations = (
        ("ase", "ase"),
        ("lammps", "lammps"),
    )

    """See LAMMPS documentation for calculator parameters.
    """

    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        # Some shared params
        command = calc_params.pop("command", "lmp")
        directory = calc_params.pop("directory", pathlib.Path.cwd())

        type_list = calc_params.pop("type_list", [])
        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i

        # eam have several formats, default to "eam"
        flavour = calc_params.pop("flavour", "eam")
        if flavour not in EAM_FLAVOURS:
            raise ValueError(f"Flavour {flavour} is not supported for EAM potential.")

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        models = canonicalise_input_models(calc_params.pop("model", []))
        self.calc_params.update(model=models)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            if models:
                calc = EAM(
                    potential=models[0],
                )
        elif self.calc_backend == "lammps":
            if models:
                pair_style = flavour
                pair_coeff = calc_params.pop("pair_coeff", "* *")
                pair_coeff += f" {models[0]} " + "{type_list}"

                calc = Lammps(
                    command=command,
                    directory=directory,
                    pair_style=pair_style,
                    pair_coeff=pair_coeff,
                    **calc_params,
                )
                # Update several params
                calc.set(units="metal", atom_style="atomic")
        else:
            ...

        self.calc = calc

        return
