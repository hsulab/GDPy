#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import itertools
import pathlib
from typing import Union, List


from ase.io import read, write


from . import AbstractPotentialManager, AbstractTrainer
from . import DummyCalculator, CommitteeCalculator


class MaceDataloader:

    name: str = "mace"

    def __init__(
        self,
        train_file: Union[str, pathlib.Path],
        test_file: Union[str, pathlib.Path],
        batchsize: int,
        directory: Union[str, pathlib.Path] = "./",
        *args,
        **kwargs,
    ) -> None:
        """"""
        self.train_file = pathlib.Path(train_file).resolve()
        self.test_file = pathlib.Path(test_file).resolve()

        self.batchsize = batchsize
        self.directory = pathlib.Path(directory).resolve()

        return

    def as_dict(
        self,
    ) -> dict:
        """"""
        params = {}
        params["name"] = self.name
        params["train_file"] = str(self.train_file)
        params["test_file"] = str(self.test_file)
        params["batchsize"] = self.batchsize
        params["directory"] = str(self.directory.resolve())

        return params


class MaceTrainer(AbstractTrainer):

    name = "mace"
    command = "mace_run_train"
    freeze_command = ""

    _train_fname = "_train.xyz"
    _test_fname = "_test.xyz"

    #: Flag indicates that the training is finished properly.
    CONVERGENCE_FLAG: str = "Done"

    def __init__(
        self,
        config: dict,
        type_list: List[str] = None,
        train_epochs: int = 200,
        print_epochs: int = 5,
        directory=".",
        command="python ./run_train.py",
        freeze_command="python ./run_train.py",
        random_seed: int = None,
        *args,
        **kwargs,
    ) -> None:
        """"""
        super().__init__(
            config,
            type_list,
            train_epochs,
            print_epochs,
            directory,
            command,
            freeze_command,
            random_seed,
            *args,
            **kwargs,
        )

        self._type_list = type_list

        return

    def _resolve_train_command(self, init_model=None, *args, **kwargs) -> str:
        """"""

        return self.command

    def freeze(self):
        """"""
        # models = list((self.directory/"checkpoints").glob("*.model"))
        use_swa = self.config.get("swa", False)
        if not use_swa:
            model_fpath = self.directory / ("{}.model".format(self.config["name"]))
        else:
            model_fpath = self.directory / ("{}_swa.model".format(self.config["name"]))

        return model_fpath

    def _resolve_freeze_command(self, *args, **kwargs) -> str:
        """"""

        return self.freeze_command

    @property
    def frozen_name(self):
        """"""

        return f"{self.name}.model"

    def _update_config(self, dataset, *args, **kwargs) -> dict:
        """"""
        # - update config
        train_config = copy.deepcopy(self.config)
        train_config["name"] = train_config.get("name", "mace")
        train_config["seed"] = self.random_seed
        train_config["train_file"] = str(dataset.train_file)
        train_config["valid_file"] = str(dataset.test_file)
        train_config["valid_fraction"] = 0.0
        test_file = train_config.get("test_file")
        if test_file is not None:
            train_config["test_file"] = str(pathlib.Path(test_file).resolve())
        else:
            train_config["test_file"] = str(dataset.test_file)

        # TODO: plus one to save the final checkpoint?
        train_config["max_num_epochs"] = self.train_epochs
        train_config["eval_interval"] = self.print_epochs
        train_config["batch_size"] = dataset.batchsize

        swa = train_config.get("swa", False)
        if swa:
            start_swa = train_config.get("start_swa", -1)
            if not (0 < start_swa < self.train_epochs):
                raise RuntimeError(
                    f"{start_swa = } must be smaller than {self.train_epochs = }"
                )
        else:
            ...

        # - misc
        import torch

        train_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        train_config["restart_latest"] = True

        train_config["save_cpu"] = True

        # - pop unused...
        train_config.pop("log_dir", None)
        train_config.pop("model_dir", None)
        train_config.pop("checkpoints_dir", None)
        train_config.pop("results_dir", None)

        return train_config

    def _train_from_the_restart(self, dataset, init_model) -> str:
        """Train from the restart"""
        if init_model is not None:
            raise NotImplementedError(
                f"{self.name} does not support initialising from a previous model."
            )

        def _add_command_options(command, config) -> str:
            """"""
            # - convert to command line options...
            command = command + " "
            for k, v in config.items():
                if isinstance(v, bool):
                    if v:
                        command += f"--{k}  "
                elif isinstance(v, int) or isinstance(v, float):
                    command += f"--{k}={str(v)}  "
                else:
                    command += f"--{k}='{str(v)}'  "

            return command

        train_config = self._update_config(dataset)

        if not self.directory.exists():
            command = self._train_from_the_scratch(dataset, init_model)
            if init_model is not None:
                ...
            command = _add_command_options(command, train_config)
        else:
            command = self._train_from_the_scratch(dataset, init_model)
            ckpt_dir = self.directory / "checkpoints"
            if ckpt_dir.exists():
                model_name = train_config["name"]
                ckpts = [p for p in ckpt_dir.glob(f"{model_name}*")]
                self._print(ckpts)
                ckpt_models = [c for c in ckpts if c.name.endswith(".model")]
                if len(ckpt_models) > 0:
                    ckpt_model = ckpt_models[0]
                    self._print(f"{ckpt_model =}")
                    prev_seed = int(ckpt_model.name.split("-")[-1][:-6])
                    self._print(f"{prev_seed =}")
                    train_config["seed"] = prev_seed
            else:
                # train from the scratch and no config needs update
                ...
            command = _add_command_options(command, train_config)

        return command

    def _prepare_dataset(self, dataset, *args, **kwargs):
        """Prepare a reann dataset for training.

        Currently, it only supports converting xyz dataset.

        """
        self._print(f"{dataset = }")
        # NOTE: make sure the dataset path exists, sometimes it will be access
        #       before training to create a shared dataset
        self.directory.mkdir(parents=True, exist_ok=True)

        if not isinstance(dataset, MaceDataloader):
            set_names, train_frames, test_frames, adjusted_batchsizes = (
                dataset.split_train_and_test()
            )

            # NOTE: reann does not support split-system training,
            #       so we need merge all structures into one List
            train_frames = itertools.chain(*train_frames)
            write(self.directory / self._train_fname, train_frames)

            test_frames = itertools.chain(*test_frames)
            write(self.directory / self._test_fname, test_frames)

            dataset = MaceDataloader(
                train_file=self.directory / self._train_fname,
                test_file=self.directory / self._test_fname,
                directory=self.directory,
                batchsize=dataset.batchsize,
            )
        else:
            ...

        return dataset

    def write_input(self, dataset, *args, **kwargs):
        """Convert dataset to the target format and write the configuration file if it has."""
        self._print(f"write {self.name} inputs...")

        # - convert dataset
        dataset = self._prepare_dataset(dataset)

        # - input config
        #   mace uses command line options

        return

    def read_convergence(self):
        """"""
        super().read_convergence()

        converged = False
        logs = list((self.directory / "logs").glob("*.log"))
        assert len(logs) == 1, "There should be only one log."
        with open(logs[0], "r") as fopen:
            lines = fopen.readlines()
        if self.CONVERGENCE_FLAG in lines[-1]:
            converged = True

        return converged


class MaceManager(AbstractPotentialManager):

    name = "mace"
    implemented_backends = ["ase"]

    valid_combinations = (("ase", "ase"),)

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
        self.calc_params.update(model=models)

        precision = calc_params.pop("precision", "float32")

        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        # - create specific calculator
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            # return ase calculator
            try:
                import torch
                from mace.calculators import MACECalculator
            except:
                raise ModuleNotFoundError(
                    "Please install mace and torch to use the ase interface."
                )
            calcs = []
            for m in models:
                # print("device", torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu")))
                curr_calc = MACECalculator(
                    model_paths=m,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    default_dtype=precision,
                )
                calcs.append(curr_calc)
            if len(calcs) == 1:
                calc = calcs[0]
            elif len(calcs) > 1:
                if estimate_uncertainty:
                    calc = CommitteeCalculator(calcs)
                else:
                    calc = calcs[0]
            else:
                ...
        elif self.calc_backend == "lammps":
            raise RuntimeError("The LAMMPS backend for MACE is under development.")
        else:
            ...

        self.calc = calc

        return

    def switch_uncertainty_estimation(self, status: bool = True):
        """Switch on/off the uncertainty estimation."""
        # NOTE: Sometimes the manager loads several models and supports uncertainty
        #       by committee but the user disables it. We need change the calc to
        #       the correct one as the loaded one is just a single calculator.
        if not hasattr(self, "calc"):
            raise RuntimeError(
                "Fail to switch uncertainty status as it does not have a calc."
            )
        # print(f"{self.calc}")

        # NOTE: make sure manager.as_dict() can have correct param
        self.calc_params["estimate_uncertainty"] = status

        # - convert calculator
        if self.calc_backend == "ase":
            if status:
                if isinstance(self.calc, CommitteeCalculator):
                    ...  # nothing to do
                else:  # reload models
                    self.register_calculator(self.calc_params)
            else:
                if isinstance(self.calc, CommitteeCalculator):
                    # TODO: save previous calc?
                    self.calc = self.calc.calcs[0]
                else:
                    ...
        elif self.calc_backend == "lammps":
            ...
        else:
            # TODO:
            # Other backends cannot have uncertainty estimation,
            # give a warning?
            ...

        return

    def remove_loaded_models(self, *args, **kwargs):
        """Loaded TF models should be removed before any copy.deepcopy operations."""
        self.calc.reset()
        if self.calc_backend == "ase":
            if isinstance(self.calc, CommitteeCalculator):
                for c in self.calc.calcs:
                    c.models = None
            else:
                self.calc.models = None
        else:
            ...

        return


if __name__ == "__main__":
    ...
