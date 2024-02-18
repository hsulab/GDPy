#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import pathlib
from typing import List


from ase.io import read, write


from . import AbstractPotentialManager, AbstractTrainer, DummyCalculator
from gdpx.computation.mixer import CommitteeCalculator


class MaceTrainer(AbstractTrainer):

    name = "mace"
    command = "python ./run_train.py"
    freeze_command = "python ./run_train.py"

    _train_fname = "_train.xyz"
    _test_fname = "_test.xyz"

    #: Flag indicates that the training is finished properly.
    CONVERGENCE_FLAG: str = "Done"

    def __init__(
        self, config: dict, type_list: List[str] = None, train_epochs: int = 200, 
        directory=".", command="python ./run_train.py", freeze_command="python ./run_train.py", 
        random_seed: int = None, *args, **kwargs
    ) -> None:
        """"""
        super().__init__(config, type_list, train_epochs, directory, command, freeze_command, random_seed, *args, **kwargs)

        self._type_list = type_list

        return
    
    def _update_config(self, dataset, *args, **kwargs):
        """"""
        super()._update_config(dataset, *args, **kwargs)
        
        # - update config
        train_config = copy.deepcopy(self.config)
        train_config["name"] = train_config.get("name", "mace")
        train_config["seed"] = self.random_seed
        train_config["train_file"] = self.directory/self._train_fname
        train_config["valid_file"] = self.directory/self._test_fname
        train_config["valid_fraction"] = 0.
        test_file = train_config.get("test_file")
        if test_file is not None:
            train_config["test_file"] = pathlib.Path(test_file).resolve()
        else:
            train_config["test_file"] = self.directory/self._test_fname

        # TODO: plus one to save the final checkpoint?
        train_config["max_num_epochs"] = self.train_epochs

        train_config["batch_size"] = dataset.batchsize

        import torch
        train_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        train_config["restart_latest"] = True

        # - pop unused...
        train_config.pop("log_dir", None)
        train_config.pop("model_dir", None)
        train_config.pop("checkpoints_dir", None)
        train_config.pop("results_dir", None)

        self.config = train_config

        return
    
    def _resolve_train_command(self, init_model=None, *args, **kwargs) -> str:
        """"""
        super()._resolve_train_command(*args, **kwargs)

        # - convert to command line options...
        command = self.command + " "
        for k, v in self.config.items():
            if isinstance(v, bool):
                if v:
                    command += f"--{k}  "
            elif isinstance(v, int) or isinstance(v, float):
                command += f"--{k}={str(v)}  "
            else:
                command += f"--{k}='{str(v)}'  "

        return command
    
    def freeze(self):
        """"""
        #models = list((self.directory/"checkpoints").glob("*.model"))
        use_swa = self.config.get("swa", False)
        if not use_swa:
            model_fpath = self.directory/("{}.model".format(self.config["name"]))
        else:
            model_fpath = self.directory/("{}_swa.model".format(self.config["name"]))

        return model_fpath
    
    def _resolve_freeze_command(self, *args, **kwargs):
        """"""
        super()._resolve_freeze_command(*args, **kwargs)

        return 

    @property
    def frozen_name(self):
        """"""

        return f"{self.name}.model"
    
    def write_input(self, dataset, *args, **kwargs):
        """Convert dataset to the target format and write the configuration file if it has."""
        super().write_input(dataset, *args, **kwargs)

        (
            set_names, train_frames, test_frames, adjusted_batchsizes
        ) = dataset.split_train_test(reduce_system=True)

        write(self.directory/self._train_fname, train_frames)
        write(self.directory/self._test_fname, test_frames)

        return
    
    def read_convergence(self):
        """"""
        super().read_convergence()

        converged = False
        logs = list((self.directory/"logs").glob("*.log"))
        assert len(logs) == 1, "There should be only one log."
        with open(logs[0], "r") as fopen:
            lines = fopen.readlines()
        if self.CONVERGENCE_FLAG in lines[-1]:
            converged = True

        return converged


class MaceManager(AbstractPotentialManager):

    name = "mace"
    implemented_backends = ["ase"]

    valid_combinations = (
        ("ase", "ase"),
    )

    def __init__(self):
        """"""

        return
    
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        # - parse params
        calc_params = copy.deepcopy(calc_params)

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", pathlib.Path.cwd())
        type_list = calc_params.pop("type_list", [])

        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i
        
        # - model files
        model_ = calc_params.get("model", [])
        if not isinstance(model_, list):
            model_ = [model_]

        models = []
        for m in model_:
            m = pathlib.Path(m).resolve()
            if not m.exists():
                raise FileNotFoundError(f"Cant find model file {str(m)}")
            models.append(str(m))

        precision = calc_params.pop("precision", "float32")

        # - create specific calculator
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            # return ase calculator
            try:
                import torch
                from mace.calculators import MACECalculator
            except:
                raise ModuleNotFoundError("Please install mace and torch to use the ase interface.")
            calcs = []
            for m in models:
                #print("device", torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu")))
                curr_calc = MACECalculator(
                    model_path=m, 
                    device=torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu")),
                    default_dtype=precision
                )
                calcs.append(curr_calc)
            if len(calcs) == 1:
                calc = calcs[0]
            elif len(calcs) > 1:
                calc = CommitteeCalculator(calcs)
            else:
                ...
        elif self.calc_backend == "lammps":
            raise RuntimeError("The LAMMPS backend for MACE is under development.")
        else:
            ...
        
        self.calc = calc

        return


if __name__ == "__main__":
    ...
