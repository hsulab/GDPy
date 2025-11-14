#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import os
import pathlib

from gdpx.backend.ase import DummyCalculator

from .manager import BasePotentialManager
from .utils import canonicalise_input_models


class FairChemManager(BasePotentialManager):

    name = "fairchem"

    implemented_backends = ("ase",)

    valid_combinations = (("ase", "ase"),)

    def register_calculator(self, calc_params: dict, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params=calc_params, *agrs, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        models = calc_params.pop("model", [])
        models = canonicalise_input_models(models, must_exist=False)
        self.calc_params.update(model=models)

        # We need set FAIRCHEM_CACHE_DIR before importing fairchem
        num_models = len(models)
        if num_models > 0:
            fake_model_fpath = pathlib.Path(models[0])
            fairchem_cache_dir, fake_model_name = (fake_model_fpath.parent, fake_model_fpath.stem)
            prev_fairchem_cache_dir = os.environ.get("FAIRCHEM_CACHE_DIR", None)
            os.environ["FAIRCHEM_CACHE_DIR"] = str(fairchem_cache_dir)
        else:
            prev_fairchem_cache_dir = None
            fake_model_name = ""

        # Some parameters for pretrained models
        head = calc_params.get("head", None)
        if head is None:
            # UMA needs head as task_name but esen does not
            # raise Exception("Please specify the task_name head for fairchem model.")
            ...

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch
                from fairchem.core import FAIRChemCalculator, pretrained_mlip
            except:
                raise ModuleNotFoundError("Please install fairchem to use the ase interface.")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            num_models = len(models)
            if num_models > 0:
                predictor = pretrained_mlip.get_predict_unit(
                    model_name=fake_model_name,
                    device=device,
                )
                calc = FAIRChemCalculator(predictor, task_name=head)
        else:
            ...

        # Restore previous FAIRCHEM_CACHE_DIR
        if prev_fairchem_cache_dir is not None:
            os.environ["FAIRCHEM_CACHE_DIR"] = str(prev_fairchem_cache_dir)

        self.calc = calc

        return


if __name__ == "__main__":
    ...
