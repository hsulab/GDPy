import pathlib
from collections.abc import Mapping

from gdpx.analysis.validators.factory import canonicalise_validator
from gdpx.analysis.validators.validator import BaseValidator
from gdpx.data.array import AtomsNDArray
from gdpx.data.loaders.dataset import AbstractDataloader
from gdpx.execution import Runtime
from gdpx.execution.factory import create_worker
from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.workflow.session.variable import DummyVariable, Variable


@registers.variable.register
class ValidatorVariable(Variable):
    def __init__(self, directory: str | pathlib.Path = "./", **kwargs):
        """Construct a validator resource."""
        # Instantiate a validator
        if isinstance(kwargs.get("worker"), Variable):
            kwargs["worker"] = kwargs["worker"].value
        validator = canonicalise_validator(kwargs)

        # Save the instance
        super().__init__(initial_value=validator, directory=directory)


@registers.operation.register
class validate(Operation):
    """The operation to validate properties by potentials.

    The reference properties should be stored and accessed through `structures`.

    """

    def __init__(
        self,
        structures,
        validator,
        runtime=None,
        run_params: dict | None = None,
        directory: str | pathlib.Path = "./",
    ) -> None:
        """Init a validate operation.

        Args:
            structures: A node that forwards structures.
            validator: A validator.
            runtime: A runtime used to create the validation worker.

        """
        runtime = DummyVariable() if runtime is None else runtime
        super().__init__(input_nodes=[structures, validator, runtime], directory=directory)

        self.run_params = run_params or {}

    def _preprocess_input_nodes(self, input_nodes):
        """Preprocess valid input nodes.

        Some arguments accept basic python objects such list or dict, which are
        not necessary to be a Variable or an Operation.

        """
        structures, validator, runtime = input_nodes

        if isinstance(validator, Mapping):
            validator = ValidatorVariable(self.directory / "validator", **validator)
            self._print(validator)

        return structures, validator, runtime

    def _convert_dataset(self, structures):
        """Validator can accept various formats of input structures.

        In a repeated workflow, the dataset is dynamic, thus,
        we need load the dataset before run...
        Validator accepts dict(reference=[], prediction=[])

        """
        dataset_ = {}
        if hasattr(structures, "items"):  # check if the input is a dict-like object
            stru_dict = structures
        else:  # assume it is just an AtomsNDArray
            stru_dict = {}
            if isinstance(structures, list):
                structures = AtomsNDArray(structures)
            stru_dict["reference"] = structures

        for k, v in stru_dict.items():
            if isinstance(v, (dict, list, AtomsNDArray)):
                ...
            elif isinstance(v, AbstractDataloader):
                v = v.load_frames()
            else:
                raise TypeError(f"{k} Dataset {type(v)} is not a dict or loader.")
            dataset_[k] = v

        dataset = dataset_

        return dataset

    def forward(self, structures, validator: BaseValidator, runtime_value):
        """Run a validator on input dataset.

        Args:
            structures: Any format that has Atoms objects.

        """
        super().forward()

        # Get a worker if the validator requires some computations
        if runtime_value is None:
            worker = None
        else:
            if isinstance(runtime_value, Runtime):
                runtime = runtime_value
            elif isinstance(runtime_value, (list, tuple)) and len(runtime_value) == 1:
                runtime = runtime_value[0]
            else:
                raise TypeError("validate requires exactly one runtime.")
            worker = create_worker(runtime, directory=self.directory)

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

    def report_convergence(self) -> bool:
        """Report convergence through the configured validator."""
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
