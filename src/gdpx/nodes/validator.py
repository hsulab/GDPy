import pathlib
from typing import Mapping, Union

from gdpx.core.register import registers
from gdpx.data.array import AtomsNDArray
from gdpx.dataloader.dataset import AbstractDataloader
from gdpx.session.operation import Operation
from gdpx.session.variable import DummyVariable, Variable
from gdpx.validator.validator import BaseValidator


@registers.variable.register
class ValidatorVariable(Variable):
    def __init__(self, directory: Union[str, pathlib.Path] = "./", **kwargs):
        """"""
        # Instantiate a validator
        method = kwargs.pop("method", "minima")
        validator = registers.create("validator", method, convert_name=False, **kwargs)

        # Save the instance
        super().__init__(initial_value=validator, directory=directory)

        return


@registers.operation.register
class validate(Operation):
    """The operation to validate properties by potentials.

    The reference properties should be stored and accessed through `structures`.

    """

    def __init__(
        self,
        structures,
        validator,
        worker=DummyVariable(),
        run_params: dict = {},
        directory: Union[str, pathlib.Path] = "./",
    ) -> None:
        """Init a validate operation.

        Args:
            structures: A node that forwards structures.
            validator: A validator.
            worker: A worker to run calculations.

        """
        super().__init__(input_nodes=[structures, validator, worker], directory=directory)

        self.run_params = run_params

        return

    def _preprocess_input_nodes(self, input_nodes):
        """Preprocess valid input nodes.

        Some arguments accept basic python objects such list or dict, which are
        not necessary to be a Variable or an Operation.

        """
        structures, validator, worker = input_nodes

        # if isinstance(validator, dict) or isinstance(validator, omegaconf.dictconfig.DictConfig):
        if isinstance(validator, Mapping):
            validator = ValidatorVariable(self.directory / "validator", **validator)
            self._print(validator)

        return structures, validator, worker

    def _convert_dataset(self, structures):
        """Validator can accept various formats of input structures.

        In an active session, the dataset is dynamic, thus,
        we need load the dataset before run...
        Validator accepts dict(reference=[], prediction=[])

        """
        dataset_ = {}
        if hasattr(structures, "items"):  # check if the input is a dict-like object
            stru_dict = structures
        else:  # assume it is just an AtomsNDArray
            stru_dict = {}
            stru_dict["reference"] = structures

        for k, v in stru_dict.items():
            if isinstance(v, dict):
                ...
            elif isinstance(v, list):
                ...
            elif isinstance(v, AtomsNDArray):
                ...
            elif isinstance(v, AbstractDataloader):
                v = v.load_frames()
            else:
                raise RuntimeError(f"{k} Dataset {type(v)} is not a dict or loader.")
            dataset_[k] = v

        dataset = dataset_

        return dataset

    def forward(self, structures, validator: BaseValidator, workers):
        """Run a validator on input dataset.

        Args:
            structures: Any format that has Atoms objects.

        """
        super().forward()

        # Get a worker if the validator requires some computations
        if workers is not None:
            num_workers = len(workers)
            assert num_workers == 1, f"Validator only accepts one worker but {num_workers} were given."
            worker = workers[0]
            worker.directory = self.directory
        else:
            worker = None

        # Convert the input structures to an object with a proper format
        dataset = self._convert_dataset(structures)

        # Run the validation
        validator.directory = self.directory
        status = validator.run(dataset, worker, **self.run_params)
        if status is None:
            status = "unfinished"
        else:
            assert isinstance(status, bool)
            if status:
                status = "finished"
            else:
                status = "unfinished"

        self.status = status

        return

    def report_convergence(self) -> bool:
        """"""
        input_nodes = self.input_nodes
        assert hasattr(input_nodes[1], "output"), (
            f"Operation {self.directory.name} cannot report convergence without forwarding."
        )
        validator = input_nodes[1].output

        self._print(f"{validator.__class__.__name__} Convergence")
        if hasattr(validator, "report_convergence"):
            converged = validator.report_convergence()
        else:
            self._print("    >>> True  (No report available)")
            converged = True

        return converged
