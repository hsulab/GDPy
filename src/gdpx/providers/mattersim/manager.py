#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from gdpx.providers.ase.backend import DummyCalculator

from ..manager_base import BasePotentialManager
from ..potential_utils import canonicalise_input_models


_PRETRAINED_MODELS = {
    "mattersim-v1.0.0-1m",
    "mattersim-v1.0.0-1m.pth",
    "mattersim-v1.0.0-5m",
    "mattersim-v1.0.0-5m.pth",
}


def canonicalise_mattersim_model(model: str) -> str:
    """Preserve MatterSim model aliases and resolve local checkpoint paths."""
    if not isinstance(model, str) or not model:
        raise ValueError("MatterSim model must be a non-empty name or checkpoint path.")
    if model.lower() in _PRETRAINED_MODELS:
        return model
    return canonicalise_input_models(model)[0]


class MatterSimManager(BasePotentialManager):

    name = "mattersim"

    implemented_backends = (
        "ase",
        "graph_pes",
    )

    valid_combinations = (
        ("ase", "ase"),
        ("graph_pes", "ase"),
    )

    def register_calculator(self, calc_params: dict, *agrs, **kwargs):
        """Register the calculator."""
        super().register_calculator(calc_params=calc_params, *agrs, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        model = canonicalise_mattersim_model(
            calc_params.pop("model", "MatterSim-v1.0.0-1M")
        )
        self.calc_params.update(model=[model])

        # Whether compute stress
        compute_stress = calc_params.pop("compute_stress", True)

        # Set the default device and update it when torch is available.
        device = calc_params.pop("device", "cpu")

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch
                from mattersim.forcefield import MatterSimCalculator

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                raise ModuleNotFoundError("Please install mattersim and torch to use the ase interface.")
            calc = MatterSimCalculator.from_checkpoint(
                load_path=model, compute_stress=compute_stress, device=device
            )
        elif self.calc_backend == "graph_pes":
            try:
                import torch
                from graph_pes.interfaces import mattersim
                from graph_pes.utils.calculator import GraphPESCalculator

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                raise ModuleNotFoundError("Please install mattersim and graph_pes to use the graph_pes interface.")

            calc = GraphPESCalculator(mattersim(load_path=model), device=device, skin=1.0)
        else:
            ...  # Backend has already been checked.

        self.calc = calc

        return


if __name__ == "__main__":
    ...
